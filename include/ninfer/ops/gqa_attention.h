#pragma once

#include "core/kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct GqaExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
};

/**
 * Returns transient arena capacity for the fused attention routes. It is nonzero for the
 * registered small-T (T=1..6) split-KV paths and zero for prompt attention.
 */
[[nodiscard]] std::size_t gqa_attention_workspace_bytes(std::int32_t q_heads, std::int32_t tokens);

/**
 * A1: append K/V at the supplied absolute positions and compute causal grouped-query attention.
 * For query head h, kvh=floor(h/group), p=positions[t], and populated cache history [0,p]:
 *
 *   score[j]    = scale * dot(q[:,h,t], K_cache[:,j,kvh]), 0 <= j <= p
 *   probability = softmax_j(score)
 *   out[:,h,t]  = BF16(sum_j probability[j] * V_cache[:,j,kvh]).
 *
 * The registered geometries are `[256,24|4,T]` group 6 and `[256,16|2,T]` group 8. q/k/v/out
 * are contiguous BF16, positions is contiguous sequential I32 [T], and scale is 1/sqrt(256).
 * Cache storage is BF16 or INT8-G64. The caller guarantees that every row in the causal domain is
 * populated and that `positions[T-1]+1` lies in the declared execution envelope. The envelope is
 * a host launch-resource promise; it does not alter the causal mask.
 *
 * q/k/v/positions/out, every cache plane, and live workspace suballocations are pairwise
 * non-overlapping. The Op overwrites every addressed cache row but owns no persistent frontier.
 */
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, KVCacheLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * A2: perform only the cache-write part of A1. k/v are contiguous BF16 `[256,4|2,T]`, positions is
 * contiguous sequential I32 [T], and every addressed code and INT8 scale is overwritten. It reads
 * no unrelated cache row, receives no execution envelope, and owns no persistent frontier.
 */
void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   KVCacheLayerView cache, cudaStream_t stream);

/**
 * A3: compute causal attention from an already populated cache without accepting new K/V or
 * mutating any cache plane. q/out are contiguous BF16 `[256,24|16,T]`, positions is contiguous
 * sequential I32 [T], and the mathematical formula and execution-envelope contract are identical
 * to A1. Small-T uses caller workspace reported by gqa_attention_workspace_bytes().
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const KVCacheLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
