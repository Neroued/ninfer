#pragma once

// qus::kernels - phase-split grouped-query full attention with folded KV append.

#include "qus/core/kv_cache.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

inline constexpr int kGqaDecodeSplits = 192;

void gqa_attention_prefill(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
                           KVCache& kv, int layer, Tensor& out, cudaStream_t stream);

void gqa_attention_decode(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& pos,
                          float scale, KVCache& kv, int layer, WorkspaceArena& ws, Tensor& out,
                          cudaStream_t stream);

} // namespace qus::kernels
