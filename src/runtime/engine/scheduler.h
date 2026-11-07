#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

// Which budget a prefill cycle runs under. A cycle is the span between two begin_prefill_cycle()
// calls; the Engine runs one Decode round per cycle.
//  - Throughput: no lane is waiting for a decode round, so prefill is bounded only by the
//    throughput token and unit allowance (the Session's pack capacity), never by time.
//  - Interleave: at least one lane waits for the next decode round, so the prefill work in front
//    of it is bounded by a token allowance and by a predicted wall-clock allowance.
enum class PrefillRegime : std::uint8_t { Throughput, Interleave };

// Per-cycle prefill allowance. The Engine derives it once (EngineCore::prefill_pacing); the
// Scheduler only applies it.
struct PrefillPacing {
    // Throughput cycle: tokens and units (a packed segment counts as one unit) that may start.
    std::uint32_t throughput_tokens_per_cycle = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t throughput_units_per_cycle  = std::numeric_limits<std::uint32_t>::max();
    // Interleave cycle: wall-clock allowance and token allowance of the prefill work between two
    // consecutive decode rounds.
    std::uint64_t interleave_ns               = std::numeric_limits<std::uint64_t>::max();
    std::uint32_t interleave_tokens_per_cycle = std::numeric_limits<std::uint32_t>::max();
    // Interleave tokens for a lane that has no cost estimate yet (first unit after binding).
    std::uint32_t interleave_cold_tokens = std::numeric_limits<std::uint32_t>::max();
    // A unit sized by the time allowance is a multiple of this; below it the allowance is spent
    // on one granule only when the unit is the first of its cycle. Truncation by the token
    // allowance or by a lane's own limit is exact and may produce any length.
    std::uint32_t unit_granularity = 1;
    // Prefill debt. While a lane waits for a decode round, every unit that ran adds its
    // wall-clock time to a debt, and every Interleave cycle repays `interleave_repay_ns` of it.
    // A cycle that starts with more than `interleave_debt_cap_ns` of debt runs the decode round
    // alone. The cap is the prefill burst a short request is let through at once; the repayment
    // is the share of time prefill keeps once a long prefill has used the burst. The default cap
    // never holds a cycle.
    std::uint64_t interleave_debt_cap_ns = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t interleave_repay_ns    = 0;
};

// Tokens of the next prefill/replay unit of each lane, indexed by lane. As an input
// (`limits`) it is the most a lane could process: the Engine fills it from its prefill chunk and
// the lane's remaining suffix, and 0 marks a lane that takes no part. As an output
// (`prefill_unit_caps`) it is the upper bound the unit may process; 0 means the allowance
// admits no unit for that lane in this cycle.
using PrefillUnitTokens = std::array<std::uint32_t, kMaximumConcurrency>;

struct PrefillMembership {
    std::array<std::uint32_t, kMaximumConcurrency> lanes{};
    std::size_t size = 0;

    [[nodiscard]] bool empty() const noexcept { return size == 0; }

    [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
        return {lanes.data(), size};
    }
};

// Membership and fairness only. Native stores own every physical reservation.
template <class Request>
class Scheduler {
public:
    using RequestPtr        = std::shared_ptr<Request>;
    using SequenceHandle    = typename Request::SequenceHandle;
    using PrefillPacing     = ::ninfer::runtime::PrefillPacing;
    using PrefillRegime     = ::ninfer::runtime::PrefillRegime;
    using PrefillUnitTokens = ::ninfer::runtime::PrefillUnitTokens;
    using PrefillMembership = ::ninfer::runtime::PrefillMembership;

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ControlMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<SequenceHandle, kMaximumConcurrency> sequences{};
        std::vector<TokenId> tokens;
        std::uint32_t row_stride = 0;
        std::size_t size         = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return {sequences.data(), size};
        }
    };

    template <class Slots>
    [[nodiscard]] RoundMembership build_round_membership(const Slots& slots,
                                                         std::uint32_t max_concurrency) const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_decode_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            if (!request->sequence) {
                throw std::logic_error("decode-ready request has no sequence handle");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.budgets[membership.size]   = RoundBudget{
                  .generated_tokens_remaining = request->recovery_pending
                                                    ? 1U
                                                    : request->output.model_token_budget_remaining(
                                                        request->budget->remaining()),
            };
            if (membership.budgets[membership.size].generated_tokens_remaining == 0) {
                throw std::logic_error("decode-ready request has no licensed model tokens");
            }
            ++membership.size;
        }
        return membership;
    }

    template <class Slots>
    [[nodiscard]] ControlMembership build_control_membership(const Slots& slots,
                                                             std::uint32_t max_concurrency) const {
        ControlMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            const auto& request = slots[lane];
            if (request == nullptr || !request->is_control_ready() || request->capture_pending) {
                continue;
            }
            if (!request->budget || !request->sequence) {
                throw std::logic_error("control-ready request has no generation state");
            }
            const std::span<const TokenId> control = request->output.pending_control_tokens();
            if (control.empty() || control.size() > request->budget->remaining()) {
                throw std::logic_error("control-ready request has no admissible control span");
            }
            if (membership.row_stride == 0) {
                membership.row_stride = static_cast<std::uint32_t>(control.size());
                membership.tokens.reserve(static_cast<std::size_t>(membership.row_stride) *
                                          max_concurrency);
            } else if (control.size() != membership.row_stride) {
                throw std::logic_error("compact control membership has ragged target spans");
            }
            membership.lanes[membership.size]     = lane;
            membership.sequences[membership.size] = *request->sequence;
            membership.tokens.insert(membership.tokens.end(), control.begin(), control.end());
            ++membership.size;
        }
        return membership;
    }

    // Prefill pacing contract. The Engine runs Control, then Decode, then Prefill in a cycle; the
    // Scheduler bounds how much Prefill work follows between two consecutive decode rounds.
    //  - Regime: begin_prefill_cycle(decode_waiting) fixes the regime of the cycle.
    //    decode_waiting is true when a lane that is not blocked by a capture waits for a decode
    //    or control round. Interleave then, Throughput otherwise.
    //  - Allowance: a Throughput cycle may start throughput_units_per_cycle units and
    //    throughput_tokens_per_cycle tokens. An Interleave cycle may process
    //    interleave_tokens_per_cycle tokens and spend interleave_ns of wall-clock time. Both
    //    are charged with what prefill_executed() reports, so the allowance left is
    //    prefill_tokens_remaining(), prefill_units_remaining() and prefill_ns_remaining().
    //    Units are unbounded and time is unlimited where the regime does not use them.
    //  - Caps: prefill_unit_caps(limits) hands the remaining allowance to the lanes in
    //    round-robin order from the cursor. A lane gets min(limit, tokens left); under
    //    Interleave also the tokens its predicted cost (prefill_cost_ns_per_token) fits into the
    //    time left, rounded down to unit_granularity, or interleave_cold_tokens while the lane
    //    has no estimate. The caps of one call add up to at most the allowance, and the
    //    first unit of a cycle always receives at least one granule, so a cycle makes
    //    progress whatever its lanes cost. The Engine passes a cap to the Session as the
    //    unit's token upper bound.
    //  - Admission: next_prefill() and prefill_candidates() visit eligible lanes round-robin
    //    from the cursor. A lane is admitted when its cap is non-zero and it fits what is left
    //    behind the lanes ahead of it: units, tokens and, under Interleave, predicted time. The
    //    first unit of a cycle is always admitted. The first eligible lane that does not fit
    //    ends the cycle and keeps the cursor, so the next cycle starts with it (no skipping).
    //  - Charging: prefill_executed() charges the tokens the Session actually processed and the
    //    wall-clock time of the unit, and moves the cursor past the lane. The elapsed time of
    //    a packed flight is attributed to its segments in proportion to their tokens by the
    //    caller. It also updates the lane's ns/token estimate (exponential average, the first
    //    sample taken as is). prefill_lane_released() forgets the estimate when the lane no
    //    longer belongs to the request that produced it.
    //  - Bound: under Interleave the tokens processed in a cycle never exceed
    //    interleave_tokens_per_cycle, and the predicted time of the units admitted after the
    //    first never exceeds interleave_ns. Under Throughput the tokens never exceed
    //    throughput_tokens_per_cycle.
    //  - Debt: a decode round costs a few milliseconds, but every prefill unit in front of it
    //    adds its flight time, so a lane that decodes beside a long prefill runs at a small
    //    fraction of its solo rate for as long as the prefill lasts. Under Interleave the
    //    Scheduler therefore keeps the wall-clock time of the units that ran as a debt that each
    //    Interleave cycle repays by interleave_repay_ns (see PrefillPacing). While the debt
    //    exceeds interleave_debt_cap_ns no lane receives a cap, and the cycle runs the decode
    //    round alone. A request whose prefill fits the cap is not delayed; a long one is limited
    //    to the share the repayment allows. A Throughput cycle ends the overlap and clears the
    //    debt.
    //  - Progress: with k eligible prefilling lanes, each lane advances at least one unit
    //    within k allowance cycles, because the first unit of each such cycle goes to the lane
    //    the cursor points at and the cursor passes it afterwards. Held cycles separate two
    //    allowance cycles by at most the debt of one allowance divided by interleave_repay_ns.
    // A cycle ends by allowance whether or not a decode-ready lane exists, so the Engine's
    // begin_cycle runs on schedule.
    void configure_prefill(PrefillPacing pacing) {
        pacing.throughput_tokens_per_cycle = std::max(pacing.throughput_tokens_per_cycle, 1U);
        pacing.throughput_units_per_cycle  = std::max(pacing.throughput_units_per_cycle, 1U);
        pacing.interleave_tokens_per_cycle = std::max(pacing.interleave_tokens_per_cycle, 1U);
        pacing.interleave_cold_tokens      = std::max(pacing.interleave_cold_tokens, 1U);
        pacing.unit_granularity            = std::max(pacing.unit_granularity, 1U);
        // A finite cap needs a repayment, or the first cycle over it would hold forever.
        if (pacing.interleave_debt_cap_ns != std::numeric_limits<std::uint64_t>::max())
            pacing.interleave_repay_ns = std::max<std::uint64_t>(pacing.interleave_repay_ns, 1);
        prefill_pacing_                    = pacing;
        begin_prefill_cycle(false);
    }

    [[nodiscard]] const PrefillPacing& prefill_pacing() const noexcept { return prefill_pacing_; }

    void begin_prefill_cycle(bool decode_waiting) noexcept {
        prefill_regime_ = decode_waiting ? PrefillRegime::Interleave : PrefillRegime::Throughput;
        prefill_units_started_ = 0;
        if (decode_waiting) {
            prefill_debt_ns_ -= std::min(prefill_debt_ns_, prefill_pacing_.interleave_repay_ns);
            if (prefill_debt_ns_ > prefill_pacing_.interleave_debt_cap_ns) {
                // A held cycle runs the decode round alone: no lane receives a cap.
                prefill_tokens_left_ = 0;
                prefill_ns_left_     = 0;
            } else {
                prefill_tokens_left_ = prefill_pacing_.interleave_tokens_per_cycle;
                prefill_ns_left_     = prefill_pacing_.interleave_ns;
            }
        } else {
            prefill_debt_ns_     = 0;
            prefill_tokens_left_ = prefill_pacing_.throughput_tokens_per_cycle;
            prefill_ns_left_     = std::numeric_limits<std::uint64_t>::max();
        }
    }

    [[nodiscard]] PrefillRegime prefill_regime() const noexcept { return prefill_regime_; }

    [[nodiscard]] std::uint32_t prefill_tokens_remaining() const noexcept {
        return prefill_tokens_left_;
    }

    // Units that may still start. Interleave does not count units: it returns the maximum.
    [[nodiscard]] std::uint32_t prefill_units_remaining() const noexcept {
        if (prefill_regime_ == PrefillRegime::Interleave)
            return std::numeric_limits<std::uint32_t>::max();
        return prefill_units_started_ >= prefill_pacing_.throughput_units_per_cycle
                   ? 0U
                   : prefill_pacing_.throughput_units_per_cycle - prefill_units_started_;
    }

    // Wall-clock time left under Interleave. Throughput is not time-bounded: the maximum.
    [[nodiscard]] std::uint64_t prefill_ns_remaining() const noexcept { return prefill_ns_left_; }

    // The ns/token estimate of a lane; nullopt until a unit of the lane has been reported.
    [[nodiscard]] std::optional<std::uint64_t>
    prefill_cost_ns_per_token(std::uint32_t lane) const noexcept {
        if (lane >= kMaximumConcurrency || prefill_cost_ns_[lane] == 0) return std::nullopt;
        return prefill_cost_ns_[lane];
    }

    // The most each lane's next unit may process when the lanes run back to back from the cursor
    // in this cycle. `limits[lane] == 0` leaves a lane out. A lane the allowance no longer
    // reaches gets 0, and so does every lane behind it.
    [[nodiscard]] PrefillUnitTokens
    prefill_unit_caps(const PrefillUnitTokens& limits) const noexcept {
        PrefillUnitTokens caps{};
        const bool interleave     = prefill_regime_ == PrefillRegime::Interleave;
        const std::uint64_t grain = prefill_pacing_.unit_granularity;
        std::uint64_t tokens_left = prefill_tokens_left_;
        std::uint64_t ns_left     = prefill_ns_left_;
        std::uint64_t units_left  = prefill_units_remaining();
        bool first                = prefill_units_started_ == 0;
        for (std::uint32_t offset = 0; offset < kMaximumConcurrency; ++offset) {
            const std::uint32_t lane = (prefill_cursor_ + offset) % kMaximumConcurrency;
            if (limits[lane] == 0) continue;
            if (units_left == 0 || tokens_left == 0 || (!first && interleave && ns_left == 0))
                break;
            std::uint64_t cap        = std::min<std::uint64_t>(limits[lane], tokens_left);
            const std::uint64_t cost = prefill_cost_ns_[lane];
            if (interleave && cost == 0) {
                cap = std::min<std::uint64_t>(cap, prefill_pacing_.interleave_cold_tokens);
            } else if (interleave) {
                const std::uint64_t affordable = ns_left / cost;
                if (affordable < cap) {
                    const std::uint64_t rounded = affordable - affordable % grain;
                    cap = rounded != 0 ? rounded : (first ? std::min(cap, grain) : 0);
                }
            }
            if (cap == 0) break;
            caps[lane] = static_cast<std::uint32_t>(cap);
            tokens_left -= cap;
            --units_left;
            if (interleave && cost != 0) ns_left -= std::min(ns_left, saturating_mul(cap, cost));
            first = false;
        }
        return caps;
    }

    // The lane that runs next in this cycle; nullopt when no lane is eligible or the first
    // eligible lane does not fit the allowance left.
    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    next_prefill(const Slots& slots, std::uint32_t concurrency,
                 const PrefillUnitTokens& caps) const noexcept {
        for (std::uint32_t offset = 0; offset < concurrency; ++offset) {
            const std::uint32_t lane = (prefill_cursor_ + offset) % concurrency;
            if (!prefill_eligible(slots[lane])) continue;
            return prefill_admits(lane, caps[lane], 0, 0, 0) ? std::optional<std::uint32_t>(lane)
                                                             : std::nullopt;
        }
        return std::nullopt;
    }

    // Lanes that the same rule admits when their units start back to back in cursor order,
    // each lane at most once. The first lane of the list is next_prefill()'s answer.
    template <class Slots>
    [[nodiscard]] PrefillMembership
    prefill_candidates(const Slots& slots, std::uint32_t concurrency,
                       const PrefillUnitTokens& caps) const noexcept {
        PrefillMembership membership;
        std::uint64_t charged_tokens = 0;
        std::uint64_t charged_ns     = 0;
        for (std::uint32_t offset = 0; offset < concurrency; ++offset) {
            const std::uint32_t lane = (prefill_cursor_ + offset) % concurrency;
            if (!prefill_eligible(slots[lane])) continue;
            if (!prefill_admits(lane, caps[lane], charged_tokens, charged_ns, membership.size))
                break;
            membership.lanes[membership.size++] = lane;
            charged_tokens += caps[lane];
            charged_ns += predicted_ns(lane, caps[lane]);
        }
        return membership;
    }

    // Reports a finished unit: the tokens it actually processed and the wall-clock time it took.
    // For a segment of a packed flight the caller passes its token share of the flight's time.
    void prefill_executed(std::uint32_t lane, std::uint32_t concurrency, std::uint32_t tokens,
                          std::uint64_t elapsed_ns) noexcept {
        prefill_cursor_ = concurrency == 0 ? 0U : (lane + 1U) % concurrency;
        if (prefill_units_started_ < std::numeric_limits<std::uint32_t>::max())
            ++prefill_units_started_;
        prefill_tokens_left_ = tokens >= prefill_tokens_left_ ? 0U : prefill_tokens_left_ - tokens;
        if (prefill_regime_ == PrefillRegime::Interleave) {
            prefill_ns_left_ = elapsed_ns >= prefill_ns_left_ ? 0U : prefill_ns_left_ - elapsed_ns;
            prefill_debt_ns_ = elapsed_ns > std::numeric_limits<std::uint64_t>::max() - prefill_debt_ns_
                                   ? std::numeric_limits<std::uint64_t>::max()
                                   : prefill_debt_ns_ + elapsed_ns;
        }
        if (lane < kMaximumConcurrency && tokens != 0 && elapsed_ns != 0) {
            const std::uint64_t sample =
                std::max<std::uint64_t>(1, (elapsed_ns + tokens / 2) / tokens);
            auto& cost = prefill_cost_ns_[lane];
            cost       = cost == 0 ? sample : std::max<std::uint64_t>(1, (cost + sample + 1) / 2);
        }
    }

    // The lane no longer belongs to the request whose units produced its estimate.
    void prefill_lane_released(std::uint32_t lane) noexcept {
        if (lane < kMaximumConcurrency) prefill_cost_ns_[lane] = 0;
    }

    // Queue/capacity changes start a bounded pass. Ordinary decode cycles only continue
    // an unfinished pass; they do not restart failed admission checks.
    void begin_admission_scan() noexcept { admission_scan_requested_ = true; }

    [[nodiscard]] bool admission_scan_pending() const noexcept {
        return admission_scan_requested_ || !admission_scan_remaining_.empty();
    }

    void admission_candidate_checked(std::uint64_t request_id) noexcept {
        std::erase(admission_scan_remaining_, request_id);
        if (request_id != admission_scan_head_) { scan_after_ = request_id; }
    }

    // Only a successful younger admission consumes the older request's bypass allowance.
    [[nodiscard]] std::vector<RequestPtr> fresh_candidates(const std::deque<RequestPtr>& fresh,
                                                           std::uint32_t concurrency) {
        std::vector<RequestPtr> candidates;
        if (fresh.empty()) {
            admission_scan_requested_ = false;
            admission_scan_remaining_.clear();
            return candidates;
        }
        admission_scan_head_ = fresh.front()->id;
        std::size_t limit    = fresh.size();
        for (std::size_t i = 0; i < fresh.size(); ++i) {
            if (fresh[i]->admission_bypasses >= concurrency) {
                limit = i + 1;
                break;
            }
        }
        if (admission_scan_requested_) {
            admission_scan_remaining_.clear();
            for (std::size_t i = 0; i < limit; ++i) {
                admission_scan_remaining_.push_back(fresh[i]->id);
            }
            admission_scan_requested_ = false;
        } else {
            // Cancellation and a newly exhausted bypass allowance can remove eligibility
            // while the pass is in progress. Keep tickets, not queue positions or iterators.
            std::erase_if(admission_scan_remaining_, [&](std::uint64_t id) {
                return std::none_of(fresh.begin(), fresh.begin() + limit,
                                    [id](const auto& request) { return request->id == id; });
            });
        }
        if (admission_scan_remaining_.empty()) { return candidates; }
        candidates.push_back(fresh.front());
        std::vector<RequestPtr> younger;
        for (std::size_t i = 1; i < limit; ++i) {
            if (std::find(admission_scan_remaining_.begin(), admission_scan_remaining_.end(),
                          fresh[i]->id) != admission_scan_remaining_.end()) {
                younger.push_back(fresh[i]);
            }
        }
        std::sort(younger.begin(), younger.end(),
                  [](const auto& a, const auto& b) { return a->id < b->id; });
        const auto next = std::find_if(younger.begin(), younger.end(), [&](const auto& request) {
            return request->id > scan_after_;
        });
        std::rotate(younger.begin(), next, younger.end());
        const auto count = std::min<std::size_t>(concurrency, younger.size());
        candidates.insert(candidates.end(), younger.begin(), younger.begin() + count);
        return candidates;
    }

    void admitted(const std::deque<RequestPtr>& fresh, const Request& admitted,
                  std::uint32_t concurrency) {
        for (const auto& skipped : fresh) {
            if (skipped->id == admitted.id) {
                service_order_.push_back(admitted.id);
                return;
            }
            if (skipped->admission_bypasses >= concurrency) {
                throw std::logic_error("fresh admission exceeded the FIFO bypass allowance");
            }
            ++skipped->admission_bypasses;
        }
        throw std::logic_error("admitted request is absent from the fresh queue");
    }

    // A pressure turn owns the next capacity handoff, including asynchronous reclamation,
    // binding and replay. Request identity remains independent of service order.
    [[nodiscard]] std::optional<std::uint64_t> progress_owner() const noexcept {
        if (!pressure_ || service_order_.empty()) return std::nullopt;
        return service_order_.front();
    }

    void require_progress(std::uint64_t request) {
        contended_ = true;
        if (pressure_) return;
        const auto owner = std::find(service_order_.begin(), service_order_.end(), request);
        if (owner == service_order_.end())
            throw std::logic_error("capacity waiter is absent from admitted membership");
        std::rotate(service_order_.begin(), owner, service_order_.end());
        pressure_ = true;
        reset_service();
    }

    // The Engine charges an execution once both new progress and retirement are known.
    // Replay and time spent waiting for a grant do not consume this service budget.
    void served(std::uint64_t request, std::uint64_t elapsed_ns) noexcept {
        if (progress_owner() != request) return;
        served_ = true;
        service_ns_ += std::min(elapsed_ns, maximum_service_ns - service_ns_);
    }

    void recovery_work(std::uint64_t request, std::uint64_t elapsed_ns) noexcept {
        if (progress_owner() == request)
            recovery_ns_ += std::min(elapsed_ns, maximum_service_ns - recovery_ns_);
    }

    // Called at a safe Engine boundary. Waiting and recovery never expire a turn.
    bool advance_service(bool have_paused, bool recovering) noexcept {
        if (!pressure_ || recovering || !served_) return false;
        if (service_ns_ < std::max(service_quantum_ns, recovery_ns_) && service_order_.size() > 1)
            return false;
        if (!have_paused && (!contended_ || service_order_.size() <= 1)) {
            pressure_ = false;
            begin_admission_scan();
        } else {
            std::rotate(service_order_.begin(), service_order_.begin() + 1, service_order_.end());
        }
        reset_service();
        contended_ = false;
        return true;
    }

    void finished(std::uint64_t request) noexcept {
        const bool owner = progress_owner() == request;
        std::erase(service_order_, request);
        if (owner) reset_service();
        if (service_order_.empty()) pressure_ = false;
        begin_admission_scan();
    }

    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    preemption_victim(const Slots& slots, std::uint32_t concurrency,
                      std::span<const std::uint32_t> excluded = {}) const noexcept {
        const auto owner = progress_owner();
        if (!owner) return std::nullopt;
        std::optional<std::uint32_t> youngest;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            const auto& request = slots[lane];
            if (!request || request->id == owner || request->is_materializing() ||
                request->is_model_finished() || request->recovery_pending ||
                std::find(excluded.begin(), excluded.end(), lane) != excluded.end())
                continue;
            if (!youngest || request->id > slots[*youngest]->id) youngest = lane;
        }
        return youngest;
    }

    // Grant at least one second of useful execution. An expensive restore earns up to
    // four seconds to amortize replay/transfer, while retaining a finite service bound.
    static constexpr std::uint64_t service_quantum_ns = 1'000'000'000;
    static constexpr std::uint64_t maximum_service_ns = 4 * service_quantum_ns;

private:
    PrefillPacing prefill_pacing_;
    PrefillRegime prefill_regime_ = PrefillRegime::Throughput;
    std::uint32_t prefill_cursor_ = 0;
    // Units reported finished in this cycle, and the allowance they left behind.
    std::uint32_t prefill_units_started_ = 0;
    std::uint32_t prefill_tokens_left_   = std::numeric_limits<std::uint32_t>::max();
    std::uint64_t prefill_ns_left_       = std::numeric_limits<std::uint64_t>::max();
    // Prefill time run since the last cycle without a waiting lane, less what was repaid.
    std::uint64_t prefill_debt_ns_ = 0;
    // ns/token estimate per lane; 0 = none.
    std::array<std::uint64_t, kMaximumConcurrency> prefill_cost_ns_{};

    template <class Slot>
    [[nodiscard]] static bool prefill_eligible(const Slot& request) noexcept {
        return request && (request->is_prefilling() || request->is_replaying()) &&
               !request->capture_pending;
    }

    [[nodiscard]] static std::uint64_t saturating_mul(std::uint64_t a, std::uint64_t b) noexcept {
        return a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a
                   ? std::numeric_limits<std::uint64_t>::max()
                   : a * b;
    }

    // Predicted time of a unit; 0 for a lane without an estimate, whose size is bounded by
    // interleave_cold_tokens instead.
    [[nodiscard]] std::uint64_t predicted_ns(std::uint32_t lane,
                                             std::uint32_t tokens) const noexcept {
        return saturating_mul(tokens, prefill_cost_ns_[lane]);
    }

    // `earlier_*` are the tokens, predicted time and units already placed ahead of this one in
    // the same candidate list, on top of what finished earlier in the cycle.
    [[nodiscard]] bool prefill_admits(std::uint32_t lane, std::uint32_t tokens,
                                      std::uint64_t earlier_tokens, std::uint64_t earlier_ns,
                                      std::uint64_t earlier_units) const noexcept {
        if (tokens == 0) return false;
        if (prefill_units_started_ == 0 && earlier_units == 0) return true;
        if (earlier_units >= prefill_units_remaining()) return false;
        if (earlier_tokens > prefill_tokens_left_ || tokens > prefill_tokens_left_ - earlier_tokens)
            return false;
        if (prefill_regime_ == PrefillRegime::Interleave) {
            if (prefill_ns_left_ == 0 || earlier_ns > prefill_ns_left_) return false;
            if (predicted_ns(lane, tokens) > prefill_ns_left_ - earlier_ns) return false;
        }
        return true;
    }

    std::uint64_t scan_after_          = 0;
    std::uint64_t admission_scan_head_ = 0;
    std::vector<std::uint64_t> admission_scan_remaining_;
    bool admission_scan_requested_ = false;
    std::deque<std::uint64_t> service_order_;
    bool pressure_             = false;
    bool contended_            = false;
    bool served_               = false;
    std::uint64_t service_ns_  = 0;
    std::uint64_t recovery_ns_ = 0;

    void reset_service() noexcept {
        served_      = false;
        service_ns_  = 0;
        recovery_ns_ = 0;
    }
};

} // namespace ninfer::runtime
