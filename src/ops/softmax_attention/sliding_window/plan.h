#pragma once

#include "ninfer/ops/sliding_window_attention.h"

namespace ninfer::ops::detail {

inline constexpr std::int32_t kSlidingWindowMaxSplits = 32;

enum class SlidingWindowAttentionRoute {
    Direct,
    SplitKv,
};

struct SlidingWindowAttentionPlan {
    SlidingWindowAttentionRoute route;
    std::int32_t tokens;
    std::int32_t warps;
    std::int32_t key_block;
    std::int32_t reduce_warps;
    std::int32_t split_capacity;
    std::int32_t max_context;
    std::int32_t window;
};

[[nodiscard]] SlidingWindowAttentionPlan
select_sliding_window_attention_plan(std::uint32_t window, std::int32_t tokens, std::int32_t batch,
                                     SlidingWindowAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
