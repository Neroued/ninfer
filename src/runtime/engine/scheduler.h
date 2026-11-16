#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <map>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

// Membership and fairness only. Native stores own every physical reservation.
template <class Request>
class Scheduler {
public:
    using RequestPtr     = std::shared_ptr<Request>;
    using SequenceHandle = typename Request::SequenceHandle;

    struct RoundMembership {
        std::vector<std::uint32_t> lanes;
        std::vector<SequenceHandle> sequences;
        std::vector<RoundBudget> budgets;

        explicit RoundMembership(std::uint32_t capacity) {
            lanes.reserve(capacity);
            sequences.reserve(capacity);
            budgets.reserve(capacity);
        }

        void clear() noexcept {
            lanes.clear();
            sequences.clear();
            budgets.clear();
        }

        [[nodiscard]] bool empty() const noexcept { return lanes.empty(); }

        [[nodiscard]] std::size_t size() const noexcept { return lanes.size(); }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept { return lanes; }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return sequences;
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept { return budgets; }
    };

    struct ControlMembership {
        std::vector<std::uint32_t> lanes;
        std::vector<SequenceHandle> sequences;
        std::vector<TokenId> tokens;
        std::uint32_t row_stride = 0;

        explicit ControlMembership(std::uint32_t capacity) {
            lanes.reserve(capacity);
            sequences.reserve(capacity);
        }

        void clear() noexcept {
            lanes.clear();
            sequences.clear();
            tokens.clear();
            row_stride = 0;
        }

        [[nodiscard]] bool empty() const noexcept { return lanes.empty(); }

        [[nodiscard]] std::size_t size() const noexcept { return lanes.size(); }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept { return lanes; }

        [[nodiscard]] std::span<const SequenceHandle> sequence_span() const noexcept {
            return sequences;
        }
    };

    template <class Slots, class Eligible>
    void build_round_membership(const Slots& slots, Eligible eligible,
                                RoundMembership& membership) const {
        membership.clear();
        for (std::size_t lane = 0; lane < slots.size(); ++lane) {
            const auto& request = slots[lane];
            if (!request || !request->is_decode_ready() || request->capture_pending ||
                !eligible(static_cast<std::uint32_t>(lane)))
                continue;
            if (!request->budget || !request->sequence)
                throw std::logic_error("decode-ready request has no generation state");
            const RoundBudget budget{
                .generated_tokens_remaining = request->recovery_pending
                                                  ? 1U
                                                  : request->output.model_token_budget_remaining(
                                                        request->budget->remaining()),
            };
            if (!budget.generated_tokens_remaining)
                throw std::logic_error("decode-ready request has no licensed model tokens");
            membership.lanes.push_back(static_cast<std::uint32_t>(lane));
            membership.sequences.push_back(*request->sequence);
            membership.budgets.push_back(budget);
        }
    }

    template <class Slots, class Eligible>
    void build_control_membership(const Slots& slots, Eligible eligible,
                                  ControlMembership& membership) const {
        membership.clear();
        for (std::size_t lane = 0; lane < slots.size(); ++lane) {
            const auto& request = slots[lane];
            if (!request || !request->is_control_ready() || request->capture_pending ||
                !eligible(static_cast<std::uint32_t>(lane)))
                continue;
            if (!request->budget || !request->sequence)
                throw std::logic_error("control-ready request has no generation state");
            const auto control = request->output.pending_control_tokens();
            if (control.empty() || control.size() > request->budget->remaining())
                throw std::logic_error("control-ready request has no admissible control span");
            if (!membership.row_stride) {
                membership.row_stride = static_cast<std::uint32_t>(control.size());
                membership.tokens.reserve(std::size_t(membership.row_stride) * slots.size());
            } else if (control.size() != membership.row_stride) {
                continue;
            }
            membership.lanes.push_back(static_cast<std::uint32_t>(lane));
            membership.sequences.push_back(*request->sequence);
            membership.tokens.insert(membership.tokens.end(), control.begin(), control.end());
        }
    }

    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    next_prefill(const Slots& slots, std::uint32_t concurrency) const noexcept {
        for (std::uint32_t offset = 0; offset < concurrency; ++offset) {
            const std::uint32_t lane = (prefill_cursor_ + offset) % concurrency;
            const auto& request      = slots[lane];
            if (request && (request->is_prefilling() || request->is_replaying()) &&
                !request->capture_pending) {
                return lane;
            }
        }
        return std::nullopt;
    }

    void prefill_executed(std::uint32_t lane, std::uint32_t concurrency) noexcept {
        prefill_cursor_ = (lane + 1U) % concurrency;
    }

    // Queue/capacity changes start a bounded pass. Ordinary decode cycles only continue
    // an unfinished pass; they do not restart failed admission checks.
    void begin_admission_scan() noexcept { admission_scan_requested_ = true; }

    [[nodiscard]] bool admission_scan_pending() const noexcept {
        return admission_scan_requested_ || !admission_scan_remaining_.empty();
    }

    void admission_candidate_checked(std::uint64_t request_id) noexcept {
        admission_scan_remaining_.erase(request_id);
        if (request_id != admission_scan_head_) { scan_after_ = request_id; }
    }

    static constexpr std::size_t admission_scan_batch = 8;

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
        if (admission_scan_requested_) {
            std::size_t limit = fresh.size();
            for (std::size_t i = 0; i < fresh.size(); ++i)
                if (fresh[i]->admission_bypasses >= concurrency) {
                    limit = i + 1;
                    break;
                }
            admission_scan_remaining_.clear();
            for (std::size_t i = 0; i < limit; ++i)
                admission_scan_remaining_.emplace(fresh[i]->id, fresh[i]);
            admission_scan_requested_ = false;
        }
        if (admission_scan_remaining_.empty()) return candidates;
        candidates.push_back(fresh.front());
        const auto append = [&](auto first, auto last) {
            for (auto it = first; it != last && candidates.size() <= admission_scan_batch; ++it) {
                if (it->first == admission_scan_head_) continue;
                if (auto request = it->second.lock()) candidates.push_back(std::move(request));
            }
        };
        const auto next = admission_scan_remaining_.upper_bound(scan_after_);
        append(next, admission_scan_remaining_.end());
        append(admission_scan_remaining_.begin(), next);
        return candidates;
    }

    void admitted(const std::deque<RequestPtr>& fresh, const Request& admitted,
                  std::uint32_t concurrency) {
        admission_scan_remaining_.erase(admitted.id);
        bool barrier = false;
        bool found   = false;
        for (const auto& skipped : fresh) {
            if (barrier) admission_scan_remaining_.erase(skipped->id);
            if (found) continue;
            if (skipped->id == admitted.id) {
                found = true;
                continue;
            }
            if (skipped->admission_bypasses >= concurrency) {
                throw std::logic_error("fresh admission exceeded the FIFO bypass allowance");
            }
            ++skipped->admission_bypasses;
            barrier |= skipped->admission_bypasses == concurrency;
        }
        if (!found) throw std::logic_error("admitted request is absent from the fresh queue");
        service_order_.push_back(admitted.id);
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
        finished(std::span<const std::uint64_t>(&request, 1));
    }

    // IDs are sorted; survivors retain service order.
    void finished(std::span<const std::uint64_t> requests) noexcept {
        const bool owner = std::binary_search(requests.begin(), requests.end(), progress_owner());
        std::erase_if(service_order_, [&](auto id) {
            return std::binary_search(requests.begin(), requests.end(), id);
        });
        if (owner) reset_service();
        if (service_order_.empty()) pressure_ = false;
        begin_admission_scan();
    }

    template <class Slots>
    [[nodiscard]] std::vector<std::uint32_t> preemption_candidates(const Slots& slots) const {
        std::vector<std::uint32_t> candidates;
        const auto owner = progress_owner();
        if (!owner) return candidates;
        for (std::uint32_t lane = 0; lane < slots.size(); ++lane) {
            const auto& request = slots[lane];
            if (request && request->id != owner && !request->is_materializing() &&
                !request->is_model_finished() && !request->recovery_pending)
                candidates.push_back(lane);
        }
        std::sort(candidates.begin(), candidates.end(),
                  [&](auto a, auto b) { return slots[a]->id > slots[b]->id; });
        return candidates;
    }

    template <class Slots>
    void append_prefill_candidates(const Slots& slots, std::vector<std::uint32_t>& candidates,
                                   std::uint32_t excluded) const {
        for (std::size_t offset = 0; offset < slots.size(); ++offset) {
            const auto lane     = (prefill_cursor_ + offset) % slots.size();
            const auto& request = slots[lane];
            if (lane != excluded && request &&
                (request->is_prefilling() || request->is_replaying()) && !request->capture_pending)
                candidates.push_back(lane);
        }
    }

    // Grant at least one second of useful execution. An expensive restore earns up to
    // four seconds to amortize replay/transfer, while retaining a finite service bound.
    static constexpr std::uint64_t service_quantum_ns = 1'000'000'000;
    static constexpr std::uint64_t maximum_service_ns = 4 * service_quantum_ns;

private:
    std::uint32_t prefill_cursor_      = 0;
    std::uint64_t scan_after_          = 0;
    std::uint64_t admission_scan_head_ = 0;
    std::map<std::uint64_t, std::weak_ptr<Request>> admission_scan_remaining_;
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
