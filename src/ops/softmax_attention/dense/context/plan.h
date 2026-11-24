#pragma once

#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

enum class ContextAttentionRoute {
    Direct,
    SplitKv,
};

struct ContextAttentionPlan {
    ContextAttentionRoute route = ContextAttentionRoute::SplitKv;
    std::int32_t tokens         = 0;
    std::int32_t warps          = 0;
    std::int32_t key_block      = 0;
    std::int32_t split_capacity = 0;
};

[[nodiscard]] ContextAttentionPlan
select_context_attention_plan(std::int32_t tokens, ContextAttentionExecutionEnvelope envelope);

} // namespace ninfer::ops::detail
