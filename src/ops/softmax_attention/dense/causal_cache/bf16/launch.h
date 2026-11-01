#pragma once

#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ninfer/ops/softmax_attention.h"

namespace ninfer::ops::detail {

void bf16_kv_append_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                              Tensor& out, CudaExecutionView execution);
void bf16_kv_cached_attention(const Tensor& q, const Tensor& positions, float scale,
                              const ConstPagedKVLayerView& cache,
                              CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                              Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops::detail
