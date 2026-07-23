#pragma once

#include "core/arena.h"
#include "core/kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Host execution-resource promise for bidirectional_gqa_attention.
 *
 * The device context_length scalar defines the exact mathematical context. This envelope only
 * bounds that scalar so a fixed launch can be captured and replayed without a host read.
 */
struct GqaContextExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

/**
 * Op: bidirectional grouped-query attention over persistent context and one query block
 *
 * For D=128, Hq=32, Hkv=8, group=4, query row i, query head h, and kvh=floor(h/4):
 *
 *   keys   = context K rows [0,L) followed logically by every query K row [0,T)
 *   score  = scale * dot(q[:,h,i], key[:,kvh,j])
 *   prob   = softmax over the complete logical key set
 *   out[:,h,i] = BF16(sum_j prob[j] * value[:,kvh,j])
 *
 * q/out are contiguous BF16 [128,32,T]. query_k/query_v are contiguous BF16 [128,8,T].
 * context_length is a contiguous device I32 scalar L. context is a read-only linear BF16 cache
 * with logical shape [128,capacity,8], of which [0,L) is populated. scale is 1/sqrt(128).
 *
 * There is no causal triangle: every query row attends every other query K/V row. Context and
 * query K/V remain separate physical segments and every input/cache byte is unchanged. out is the
 * only observable mutation and is completely overwritten. The current optimized implementation
 * domain is T=1..16 on sm_120a.
 *
 * The caller guarantees min_context <= L <= max_context and L <= context.max_context. The
 * execution envelope may affect finite launch selection and workspace capacity, never the
 * admitted key set or numerical result.
 */
void bidirectional_gqa_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                 const Tensor& context_length, float scale,
                                 const KVCacheLayerView& context,
                                 GqaContextExecutionEnvelope envelope, WorkspaceArena& workspace,
                                 Tensor& out, cudaStream_t stream);

/**
 * Returns the transient arena capacity required for one exact T in the optimized T=1..16 domain.
 */
[[nodiscard]] std::size_t bidirectional_gqa_attention_workspace_bytes(std::int32_t tokens);

} // namespace ninfer::ops
