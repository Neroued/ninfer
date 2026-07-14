#pragma once

#include "runtime/contract/types.h"

#include <cstdint>
#include <cstdlib>

namespace ninfer::runtime {

class GenerationBudget {
public:
    GenerationBudget(std::uint32_t effective_tokens, FinishReason limit_reason)
        : remaining_(effective_tokens), limit_reason_(limit_reason) {
        if (limit_reason_ != FinishReason::OutputLimit &&
            limit_reason_ != FinishReason::ContextCapacity) {
            std::abort();
        }
    }

    [[nodiscard]] std::uint32_t remaining() const noexcept { return remaining_; }

    [[nodiscard]] FinishReason limit_reason() const noexcept { return limit_reason_; }

    [[nodiscard]] RoundBudget round_budget() const noexcept {
        return RoundBudget{.generated_tokens_remaining = remaining_};
    }

    void commit(std::uint32_t tokens) noexcept {
        if (tokens > remaining_) { std::abort(); }
        remaining_ -= tokens;
    }

private:
    std::uint32_t remaining_   = 0;
    FinishReason limit_reason_ = FinishReason::None;
};

} // namespace ninfer::runtime
