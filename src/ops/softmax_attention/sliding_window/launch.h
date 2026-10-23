#pragma once

#include "ops/softmax_attention/sliding_window/plan.h"

namespace ninfer::ops::detail {

void launch_sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     float scale, const ConstCyclicKVCacheLayerView& context,
                                     const SlidingWindowAttentionPlan& plan, Tensor& partial_acc,
                                     Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
