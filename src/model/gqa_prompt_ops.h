#pragma once

// Model-private GQA prompt decomposition used by the MTP schedule.

#include "ninfer/core/kv_cache.h"
#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, KVCache& kv,
                   int layer, cudaStream_t stream);
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale, KVCache& kv,
                          int layer, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels
