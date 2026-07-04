#pragma once

// qus::kernels - grouped-query full attention with folded KV append.

#include "qus/core/kv_cache.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

inline constexpr int kGqaDecodeSplits = 192;

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCache& kv, int layer, WorkspaceArena& ws, Tensor& out,
                   cudaStream_t stream);

} // namespace qus::kernels
