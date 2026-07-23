#pragma once

// ninfer::ops::detail - private launch prototypes for gqa_attention policies.

#include "core/kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class GqaAttentionRoute { SmallT, ChunkedSmallT, Prompt };

std::int32_t gqa_attention_decode_splits(std::int32_t q_heads, std::int32_t kv_heads);

bool gqa_attention_uses_small_t(std::int32_t tokens);

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t tokens,
                                              GqaExecutionEnvelope envelope);

const char* gqa_attention_route_name(GqaAttentionRoute route);

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions, float scale, KVCacheLayerView cache,
                                  GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                  Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                  cudaStream_t stream);

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const KVCacheLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream);

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, float scale, KVCacheLayerView cache,
                                 Tensor& out, cudaStream_t stream);

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          KVCacheLayerView cache, cudaStream_t stream);

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const KVCacheLayerView& cache, Tensor& out,
                                           cudaStream_t stream);

void gqa_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& positions, float scale, KVCacheLayerView cache,
                          GqaExecutionEnvelope envelope, Tensor* partial_acc, Tensor* partial_m,
                          Tensor* partial_l, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
