#pragma once

#include "ninfer/types.h"
#include "runtime/contract/execution.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
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
                  std::uint32_t concurrency) const {
        for (const auto& skipped : fresh) {
            if (skipped->id == admitted.id) { return; }
            if (skipped->admission_bypasses >= concurrency) {
                throw std::logic_error("fresh admission exceeded the FIFO bypass allowance");
            }
            ++skipped->admission_bypasses;
        }
        throw std::logic_error("admitted request is absent from the fresh queue");
    }

    template <class Slots>
    [[nodiscard]] std::optional<std::uint32_t>
    youngest_victim(const Slots& slots, std::uint32_t concurrency,
                    std::optional<std::uint64_t> priority_ticket = std::nullopt,
                    std::span<const std::uint32_t> excluded      = {}) const noexcept {
        std::optional<std::uint32_t> oldest;
        std::optional<std::uint32_t> youngest;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            const auto& request = slots[lane];
            if (!request || request->is_materializing() || request->is_model_finished()) {
                continue;
            }
            if (!oldest || request->id < slots[*oldest]->id) { oldest = lane; }
            if (!request->recovery_pending &&
                std::find(excluded.begin(), excluded.end(), lane) == excluded.end() &&
                (!priority_ticket || request->id > *priority_ticket) &&
                (!youngest || request->id > slots[*youngest]->id)) {
                youngest = lane;
            }
        }
        if (!priority_ticket && youngest == oldest) { return std::nullopt; }
        return youngest;
    }

    void preempted() noexcept {
        restoration_event_ = false;
        // The paused request waits for a capacity event. Its free lane can immediately
        // serve another request that fits the remaining physical capacity.
        begin_admission_scan();
    }

    // Completion, cancellation or a committed permanent shrink can make a snapshot fit.
    // Returning speculative scratch at the end of an ordinary unit does not call this.
    void capacity_released() noexcept { restoration_event_ = true; }

    [[nodiscard]] bool should_restore(bool have_paused, bool resident_empty) const noexcept {
        return have_paused && (resident_empty || restoration_event_);
    }

    // A successful restore keeps priority through replay and its first new execution unit.
    void restoration_blocked() noexcept { restoration_event_ = false; }

    void paused_queue_empty() noexcept { restoration_event_ = false; }

private:
    std::uint32_t prefill_cursor_      = 0;
    std::uint64_t scan_after_          = 0;
    std::uint64_t admission_scan_head_ = 0;
    std::vector<std::uint64_t> admission_scan_remaining_;
    bool admission_scan_requested_ = false;
    bool restoration_event_        = false;
};

} // namespace ninfer::runtime
