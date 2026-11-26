#pragma once

#include "ninfer/ops/softmax_attention.h"
#include "ops/softmax_attention/kernels/context_query/group3_schedule.h"

namespace ninfer::ops::detail {

inline constexpr ContextBucket kContextAttentionMaxBucket = ContextBucket::M1;

enum class ContextAttentionRoute {
    Direct,
    SplitKv,
};

struct ContextAttentionPlan {
    AttentionHeadGeometry geometry;
    ContextAttentionRoute route = ContextAttentionRoute::SplitKv;
    std::int32_t tokens         = 0;
    std::int32_t key_block      = 0;
    std::int32_t split_capacity = 0;
    std::int32_t reduce_warps      = 4;
    Group3Schedule group3_schedule = Group3Schedule::Standard;
};

[[nodiscard]] ContextAttentionPlan select_context_attention_plan(AttentionHeadGeometry geometry,
                                                                 ContextBucket bucket,
                                                                 std::int32_t batch,
                                                                 std::int32_t tokens);

} // namespace ninfer::ops::detail
