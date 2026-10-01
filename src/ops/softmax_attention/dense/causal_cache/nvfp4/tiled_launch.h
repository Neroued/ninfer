#pragma once
#include "ops/softmax_attention/common/causal_partition.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"

namespace ninfer::ops::detail {
void nvfp4_kv_tiled_attention(const CausalAttentionOperands&, Nvfp4KvReadView, CausalKvPartition,
                              WorkspaceArena&, cudaStream_t);
} // namespace ninfer::ops::detail
