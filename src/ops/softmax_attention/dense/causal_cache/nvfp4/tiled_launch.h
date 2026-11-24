#pragma once
#include "ops/softmax_attention/kernels/causal/operands.h"

namespace ninfer::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands&, Nvfp4KvReadView, cudaStream_t);
} // namespace ninfer::ops::detail
