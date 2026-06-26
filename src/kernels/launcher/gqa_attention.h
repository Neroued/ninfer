#pragma once

// qus::kernels::detail - private launch prototypes for gqa_attention variants.

#include "qus/core/kv_cache.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void gqa_attention_prefill_launch(const Tensor& q, const Tensor& k, const Tensor& v, float scale,
                                  KVCache& kv, int layer, Tensor& out, cudaStream_t stream);

void gqa_attention_decode_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& pos, float scale, KVCache& kv, int layer,
                                 Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
