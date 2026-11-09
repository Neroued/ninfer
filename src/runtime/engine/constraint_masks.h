#pragma once

// Structured output (P4): the Engine side of runtime::TokenMaskProvider. capy-flash file; the
// shape follows upstream 08902f73 (EngineCore::RoundMasks).

#include "runtime/contract/execution.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <span>
#include <stdexcept>

namespace ninfer::runtime {

namespace testing {
// Test injection, set only by test programs (it replaces the NINFER_CONSTRAINT_DEAD_END_AFTER
// environment hook, which any production process could pick up): when N != 0, from the N-th
// output token of every constrained request on the language has no legal token.
inline std::atomic<std::size_t> constraint_dead_end_after{0};
// Containment tests: when N != 0, filling the masks of a constrained request fails from its N-th
// output token on (an exception out of the matcher); when true, the matcher "refuses" the
// thinking-control span of every constrained request (EngineCore::run_control_batch).
inline std::atomic<std::size_t> constraint_mask_failure_after{0};
inline std::atomic<bool> constraint_refuse_control{false};
} // namespace testing

// Maps the compact rows of one Program call to the requests' output sessions. Lives on the worker
// stack for that call only.
template <class OutputSession>
class RoundMasks final : public TokenMaskProvider {
public:
    // `committed`: output tokens the request has committed so far (test injection below).
    void bind(std::size_t row, OutputSession* output, std::size_t committed) noexcept {
        outputs_[row]   = output;
        committed_[row] = committed;
        any_            = any_ || (output != nullptr && output->constrained());
    }

    // True when at least one bound row is constrained; otherwise the Program receives no provider
    // and the round is exactly the unconstrained one.
    [[nodiscard]] bool any() const noexcept { return any_; }

    [[nodiscard]] bool constrained(std::size_t row) const noexcept override {
        return row < outputs_.size() && outputs_[row] != nullptr && outputs_[row]->constrained();
    }

    [[nodiscard]] std::uint32_t fill(std::size_t row, std::span<const TokenId> drafts,
                                     std::span<std::uint32_t> words) override {
        const std::size_t positions = drafts.size() + 1;
        const std::size_t stride    = words.size() / positions;
        std::uint32_t dead          = 0;
        try {
            if (const std::size_t after =
                    testing::constraint_mask_failure_after.load(std::memory_order_relaxed);
                after != 0 && committed_[row] + 1 >= after) {
                throw std::runtime_error("injected mask fill failure (test)");
            }
            dead = outputs_[row]->grammar_masks(drafts, words);
        } catch (const std::exception& error) {
            // A matcher failure while filling belongs to this request only: every position becomes
            // a dead end (the one-token mask a real dead end carries), so the row fails with
            // ConstraintDeadEnd at commit and the rest of the batch runs normally. The matcher
            // rolled its lookahead back before rethrowing.
            std::fprintf(stderr, "[constraint] mask fill failed, the request fails: %s\n",
                         error.what());
            for (std::size_t position = 0; position < positions; ++position) {
                const auto mask = words.subspan(position * stride, stride);
                std::fill(mask.begin(), mask.end(), 0U);
                mask[0] = 1U;
                dead |= 1U << position;
            }
            return dead;
        }
        if (const std::size_t after =
                testing::constraint_dead_end_after.load(std::memory_order_relaxed);
            after != 0) {
            // Test injection (testing::constraint_dead_end_after = N): from the N-th output token
            // on the language has no legal token, exactly as a real dead end is reported by the
            // matcher (dead bit, safe one-token mask).
            for (std::size_t position = 0; position < positions; ++position) {
                if (committed_[row] + position + 1 < after) { continue; }
                const auto mask = words.subspan(position * stride, stride);
                std::fill(mask.begin(), mask.end(), 0U);
                mask[0] = 1U;
                dead |= 1U << position;
            }
        }
        return dead;
    }

private:
    std::array<OutputSession*, kMaximumConcurrency> outputs_{};
    std::array<std::size_t, kMaximumConcurrency> committed_{};
    bool any_ = false;
};

} // namespace ninfer::runtime
