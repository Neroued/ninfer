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

// Per-cycle prefill budget. A cycle is the span between two begin_prefill_cycle() calls; the
// Engine runs one Decode round per cycle, so this budget bounds the prefill work between two
// consecutive decode rounds.
struct PrefillPolicy {
    // Unit count that may start in one cycle, including the unconditional first unit.
    std::uint32_t max_units_per_cycle = std::numeric_limits<std::uint32_t>::max();
    // Token allowance for the units that follow the first one in the same cycle.
    std::uint32_t max_tokens_per_cycle = std::numeric_limits<std::uint32_t>::max();
};

// Upper bound of tokens the next prefill/replay unit of each lane may process, indexed by
// lane. The Engine fills it from its prefill chunk and the lane's remaining suffix; entries of
// lanes that are not prefilling are ignored.
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
    using PrefillPolicy     = ::ninfer::runtime::PrefillPolicy;
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

    // Prefill cycle contract (the Engine runs Decode once per cycle and Prefill after it):
    //  - Admission: the first unit of a cycle is always admitted. A later unit is admitted
    //    only when its declared tokens fit prefill_tokens_remaining() and fewer than
    //    max_units_per_cycle units have started.
    //  - Order: lanes are visited round-robin from the cursor, which moves past a lane only
    //    when prefill_executed() reports it. The first eligible lane that does not fit ends
    //    the cycle and keeps the cursor, so the next cycle starts with it (no skipping).
    //  - Charging: prefill_executed() charges the tokens the Session actually processed.
    //  - Bound: prefill tokens executed in one cycle <= max(max_tokens_per_cycle, largest
    //    unit_tokens). Every unit_tokens <= prefill_chunk, so with a decode-ready lane the
    //    prefill work between two consecutive decode rounds is at most
    //    max(max_tokens_per_cycle, prefill_chunk).
    //  - Progress: with k eligible prefilling lanes, each lane advances at least one unit
    //    within k cycles, because the first unit of each cycle goes to the lane the cursor
    //    points at and the cursor passes it afterwards.
    // A cycle ends by budget whether or not a decode-ready lane exists, so the Engine's
    // begin_cycle runs on schedule.
    void configure_prefill(PrefillPolicy policy) {
        prefill_policy_ = policy;
        begin_prefill_cycle();
    }

    [[nodiscard]] const PrefillPolicy& prefill_policy() const noexcept { return prefill_policy_; }

    void begin_prefill_cycle() noexcept {
        prefill_units_started_ = 0;
        prefill_tokens_left_   = prefill_policy_.max_tokens_per_cycle;
    }

    [[nodiscard]] std::uint32_t prefill_tokens_remaining() const noexcept {
        return prefill_tokens_left_;
    }

    [[nodiscard]] std::uint32_t prefill_units_remaining() const noexcept {
        return prefill_units_started_ >= prefill_policy_.max_units_per_cycle
                   ? 0U
                   : prefill_policy_.max_units_per_cycle - prefill_units_started_;
    }

    // The lane that runs next in this cycle; nullopt when no lane is eligible or the first
    // eligible lane does not fit the remaining budget.
    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    next_prefill(const Slots& slots, std::uint32_t concurrency,
                 const PrefillUnitTokens& unit_tokens) const noexcept {
        for (std::uint32_t offset = 0; offset < concurrency; ++offset) {
            const std::uint32_t lane = (prefill_cursor_ + offset) % concurrency;
            if (!prefill_eligible(slots[lane])) continue;
            return prefill_admits(unit_tokens[lane], 0, 0) ? std::optional<std::uint32_t>(lane)
                                                           : std::nullopt;
        }
        return std::nullopt;
    }

    // Lanes that the same rule admits when their units start back to back in cursor order,
    // each lane at most once. The first lane of the list is next_prefill()'s answer.
    template <class Slots>
    [[nodiscard]] PrefillMembership
    prefill_candidates(const Slots& slots, std::uint32_t concurrency,
                       const PrefillUnitTokens& unit_tokens) const noexcept {
        PrefillMembership membership;
        std::uint64_t charged = 0;
        for (std::uint32_t offset = 0; offset < concurrency; ++offset) {
            const std::uint32_t lane = (prefill_cursor_ + offset) % concurrency;
            if (!prefill_eligible(slots[lane])) continue;
            if (!prefill_admits(unit_tokens[lane], charged, membership.size)) break;
            membership.lanes[membership.size++] = lane;
            charged += unit_tokens[lane];
        }
        return membership;
    }

    // Reports a finished unit with the tokens it actually processed.
    void prefill_executed(std::uint32_t lane, std::uint32_t concurrency,
                          std::uint32_t tokens) noexcept {
        prefill_cursor_ = (lane + 1U) % concurrency;
        if (prefill_units_started_ < std::numeric_limits<std::uint32_t>::max())
            ++prefill_units_started_;
        prefill_tokens_left_ = tokens >= prefill_tokens_left_ ? 0U : prefill_tokens_left_ - tokens;
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
    PrefillPolicy prefill_policy_;
    std::uint32_t prefill_cursor_ = 0;
    // Units reported finished in this cycle, and the allowance they left behind.
    std::uint32_t prefill_units_started_ = 0;
    std::uint32_t prefill_tokens_left_   = std::numeric_limits<std::uint32_t>::max();

    template <class Slot>
    [[nodiscard]] static bool prefill_eligible(const Slot& request) noexcept {
        return request && (request->is_prefilling() || request->is_replaying()) &&
               !request->capture_pending;
    }

    // `earlier_units`/`earlier_tokens` are units already placed ahead of this one in the
    // same candidate list, on top of what finished earlier in the cycle.
    [[nodiscard]] bool prefill_admits(std::uint32_t tokens, std::uint64_t earlier_tokens,
                                      std::uint64_t earlier_units) const noexcept {
        if (prefill_units_started_ == 0 && earlier_units == 0) return true;
        if (prefill_units_started_ + earlier_units >= prefill_policy_.max_units_per_cycle)
            return false;
        return earlier_tokens <= prefill_tokens_left_ &&
               tokens <= prefill_tokens_left_ - earlier_tokens;
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
