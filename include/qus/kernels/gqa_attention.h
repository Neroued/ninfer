#pragma once

// qus::kernels - grouped-query full attention with folded KV append.

#include "qus/core/kv_cache.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

// The long-context grid is four KV heads × 85 splits = 340 CTAs, matching two
// CTA slots per SM on the RTX 5090's 170 SMs.
inline constexpr int kGqaDecodeSplits = 85;

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCache& kv, int layer, WorkspaceArena& ws, Tensor& out,
                   cudaStream_t stream);

} // namespace qus::kernels
