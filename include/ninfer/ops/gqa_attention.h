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
 * Shared numerical contract for A1/A2/A3.
 *
 * Public q/k/v inputs and BF16 cache values are interpreted after their BF16 storage boundary.
 * INT8-G64 cache rows use one FP16 scale for each contiguous 64-element group. For BF16 source
 * values x, their exact observable encoding is:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s
 *
 * A1 and A2 produce identical code and scale bits. The common ideal attention oracle uses BF16 Q
 * and logical cache values (BF16 values for a BF16 cache, FP32 decode above for INT8-G64), then
 * evaluates score dot products, stable softmax, and value reduction in FP64. The BF16 Op output is
 * promoted to FP64 for comparison with that result.
 *
 * The registered INT8 implementation defines Q8-G64, paired with INT8-G64 K, as its native query
 * compute profile. Its profile-defined query quantization and any narrower staging do not replace
 * BF16 Q in the ideal oracle. BF16 and INT8 implementations therefore use the separate named
 * `attention_bf16` and `attention_int8` numerical qualification tolerances. Those envelopes apply
 * to the registered geometries, tested token extents, conformance matrix, and target-representative
 * activation range; they are not a universal error bound for arbitrary adversarial BF16 tensors.
 * A1 and A3 are each qualified directly against the ideal oracle. A1-versus-A3 parity is only an
 * additional consistency check.
 */

/**
 * Returns transient arena capacity required for the exact positive T supplied.
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
 * T may be any positive value that fits the declared cache and execution envelope.
 * Cache storage is BF16 or INT8-G64 under the shared numerical contract above. The caller
 * guarantees that every row in the causal domain is populated and that `positions[T-1]+1` lies in
 * the declared execution envelope. The envelope is a host launch-resource promise; it does not
 * alter the causal mask.
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
 * to A1. Caller workspace is reported by gqa_attention_workspace_bytes() for the exact T.
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const KVCacheLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
