#pragma once

#include "ninfer/ops/sliding_window_attention.h"
#include "ops/softmax_attention/kernels/context_query/group3_schedule.h"

namespace ninfer::ops::detail {

inline constexpr std::int32_t kSlidingWindowMaxSplits = 32;

struct SlidingWindowLaunchShape {
    AttentionHeadGeometry geometry;
    std::int32_t tokens;
    std::int32_t reduce_warps;
    std::int32_t split_capacity;
    std::int32_t max_context;
    std::int32_t window;
};

struct BlockSlidingWindowAttentionPlan : SlidingWindowLaunchShape {
    Group3Schedule schedule;
};

[[nodiscard]] SlidingWindowLaunchShape
select_sliding_window_attention_plan(AttentionHeadGeometry geometry, std::uint32_t window,
                                     ContextBucket bucket, std::int32_t batch, std::int32_t tokens);

[[nodiscard]] BlockSlidingWindowAttentionPlan
select_block_sliding_window_attention_plan(AttentionHeadGeometry geometry, std::uint32_t window,
                                           ContextBucket bucket, std::int32_t batch,
                                           std::int32_t tokens);

} // namespace ninfer::ops::detail
