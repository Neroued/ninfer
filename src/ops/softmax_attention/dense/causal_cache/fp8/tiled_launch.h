#pragma once

#include "ops/softmax_attention/kernels/causal/partition.h"
#include "ops/softmax_attention/kernels/causal/operands.h"

namespace ninfer::ops::detail {
void fp8_kv_tiled_attention(const CausalAttentionOperands&, Fp8KvReadView, CausalKvPartition,
                            MemoryArena&, cudaStream_t);
} // namespace ninfer::ops::detail
