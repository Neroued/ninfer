#pragma once

#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/operands.h"

namespace ninfer::ops::detail {
void k8v4_kv_tiled_attention(const CausalAttentionOperands&, K8V4KvReadView, CausalKvPartition,
                             MemoryArena&, cudaStream_t);
} // namespace ninfer::ops::detail
