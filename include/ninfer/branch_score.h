#pragma once

#include "ninfer/types.h"
#include <span>
#include <vector>

namespace ninfer {

// Borrowed until score_branches returns, including cancellation cleanup. The complete predictor
// is prefix || suffix. root_tokens marks the static prefix and trunk_tokens the context boundary.
// Both are token counts inside prefix; zero disables that level of reuse.
struct BranchScoreRow {
    std::span<const TokenId> prefix;
    std::span<const TokenId> suffix;
    std::span<const TokenId> candidates;
    std::uint32_t trunk_tokens = 0;
    std::uint32_t root_tokens  = 0;

    [[nodiscard]] std::size_t size() const noexcept { return prefix.size() + suffix.size(); }

    [[nodiscard]] TokenId operator[](std::size_t i) const noexcept {
        return i < prefix.size() ? prefix[i] : suffix[i - prefix.size()];
    }
};

struct BranchScoreResult {
    std::vector<std::vector<float>> logits;
    std::uint64_t computed_tokens = 0;
    std::uint64_t reused_tokens   = 0;
    std::uint64_t forward_batches = 0;
    std::uint32_t largest_batch   = 0;
    double prefill_ms             = 0;
};

} // namespace ninfer
