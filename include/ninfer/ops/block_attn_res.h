#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

// Scales and epsilons must be finite and positive.
struct BlockAttnResParameters {
    float query_scale     = 1.0f;  // Finite positive query logit scale (e.g., 1/sqrt(D) or 1.0).
    float rms_eps         = 1e-6f; // Source RMSNorm epsilon.
    float norm_eps        = 1e-6f; // Output RMSNorm epsilon.
    bool normalize_values = true;
};

/**
 * Mixes N_stack + 1 sources for each token column m and query q.
 *
 *   x[n] = FP32(stack[:,m,n])                        for n < N_stack
 *   x[N_stack] = FP32(partial[:,m]) + FP32(addend[:,m])
 *   inv_rms[n] = rsqrt(sum_d(x[n]^2) / D + params.rms_eps)
 *   logits[q,n] = params.query_scale * dot(queries[:,q], x[n]) * inv_rms[n]
 *   p[q,:] = softmax(logits[q,:])
 *   mix[:,m,q] = sum_n(p[q,n] * value[n])
 *
 * An empty addend contributes zero. The partial sum stays in FP32. With
 * normalize_values enabled, value[n] = x[n] * inv_rms[n]. Otherwise value[n] = x[n].
 *
 * A norm_weight of contiguous BF16 [D,Q] applies output RMSNorm before
 * RN_BF16 storage. Each output column is multiplied by
 * rsqrt(sum_d(mix^2) / D + params.norm_eps) and norm_weight[:,q].
 * With empty norm_weight, out stores the FP32 mix directly.
 *
 * partial and non-empty addend are contiguous BF16 [D,M]. stack is empty or
 * contiguous BF16 [D,M,N_stack]. queries is contiguous FP32 [D,Q].
 * D, M and Q are positive. out is contiguous [D,M,Q] and disjoint from all inputs.
 * All non-empty operands belong to execution.device() and require 16-byte alignment
 * when D is divisible by 8 and natural alignment otherwise.
 * Zero caller workspace. CUDA Graph capture and replay are supported.
 */
void block_attn_res(const Tensor& stack, const Tensor& partial, const Tensor& addend,
                    const Tensor& queries, const Tensor& norm_weight,
                    const BlockAttnResParameters& params, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
