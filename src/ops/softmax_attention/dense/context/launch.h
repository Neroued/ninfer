#pragma once

#include "ops/softmax_attention/dense/context/plan.h"

namespace ninfer::ops::detail {

void launch_context_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& context_lengths, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale,
                              const ConstPagedKVBatchLayerView& context,
                              const ContextAttentionPlan& plan, Tensor& partial_acc,
                              Tensor& partial_m, Tensor& partial_l, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
