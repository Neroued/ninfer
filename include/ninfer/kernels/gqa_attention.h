#pragma once

// ninfer::kernels - grouped-query full attention with folded KV append.

#include "ninfer/core/kv_cache.h"
#include "ninfer/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::kernels {

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCache& kv, int layer, WorkspaceArena& ws, Tensor& out,
                   cudaStream_t stream);

} // namespace ninfer::kernels
