#pragma once

#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

inline constexpr ContextBucket kContextAttentionMaxBucket = ContextBucket::K256;

enum class ContextAttentionRoute {
    Direct,
    SplitKv,
};

struct ContextAttentionPlan {
    ContextAttentionRoute route = ContextAttentionRoute::SplitKv;
    std::int32_t tokens         = 0;
    std::int32_t key_block      = 0;
    std::int32_t split_capacity = 0;
};

[[nodiscard]] ContextAttentionPlan
select_context_attention_plan(ContextBucket bucket, std::int32_t batch, std::int32_t tokens);

} // namespace ninfer::ops::detail
