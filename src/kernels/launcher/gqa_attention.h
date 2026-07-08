#pragma once

// qus::kernels::detail - private launch prototypes for gqa_attention policies.

#include "qus/core/kv_cache.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace qus::kernels::detail {

bool gqa_attention_uses_small_t(std::int32_t tokens);

void gqa_attention_kv_quantize_append_launch(const Tensor& k, const Tensor& v,
                                             const Tensor& positions, KVCache& kv, int layer,
                                             bool positions_are_base, cudaStream_t stream);

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, float scale, KVCache& kv, int layer,
                                  Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                  Tensor& out, cudaStream_t stream);

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, float scale, KVCache& kv, int layer,
                                 Tensor& out, cudaStream_t stream);

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           KVCache& kv, int layer, Tensor& out,
                                           cudaStream_t stream);

void gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& positions, float scale, KVCache& kv, int layer,
                          Tensor* partial_acc, Tensor* partial_m, Tensor* partial_l, Tensor& out,
                          cudaStream_t stream);

} // namespace qus::kernels::detail
