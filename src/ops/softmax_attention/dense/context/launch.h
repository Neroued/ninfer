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
context_attention_resolve_plan(std::int32_t tokens, ContextAttentionExecutionEnvelope envelope);

[[nodiscard]] const char* context_attention_route_name(ContextAttentionRoute route);

void context_attention_launch(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& context_lengths, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale,
                              const PagedKVBatchLayerView& context,
                              const ContextAttentionPlan& plan, Tensor& partial_acc,
                              Tensor& partial_m, Tensor& partial_l, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
