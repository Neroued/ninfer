#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * Apply plain per-head RMSNorm followed by full-head split-half 1-D RoPE in place.
 *
 * The pair profile is q BF16 [128,32,W,B], k BF16 [128,8,W,B], q_norm_weight and
 * k_norm_weight BF16 [128], and positions I32 [W,B], with W=2..16 and B=1..8. For each head and
 * physical token, the complete mathematical operation is
 *
 *   inv       = 1 / sqrt(sum_d x[d]^2 / 128 + 1e-6)
 *   n[d]      = x[d] * inv * norm_weight[d]
 *   angle(i)  = position * (1e7)^(-2*i/128), 0<=i<64
 *   out[i]    = n[i]    * cos(angle(i)) - n[i+64] * sin(angle(i))
 *   out[i+64] = n[i+64] * cos(angle(i)) + n[i]    * sin(angle(i)).
 *
 * Normalization and rotation form one semantic operation: there is no observable BF16
 * materialization of n. q and k are completely overwritten in place and must not overlap each
 * other, positions, or either norm weight. Read-only norm weights may overlap each other.
 * positions and weights remain unchanged. All tensors are contiguous and 4-byte aligned.
 * The Op owns no workspace or persistent state.
 */
void rmsnorm_rope(const Tensor& positions, const Tensor& q_norm_weight, const Tensor& k_norm_weight,
                  Tensor& q, Tensor& k, CudaExecutionView execution);

/**
 * Single-K form of the same formula and effects. x is BF16 [128,8,T], norm_weight is BF16 [128],
 * and positions is I32 [T], with T=1..2048. x is completely overwritten in place and must not
 * overlap either read-only input.
 */
void rmsnorm_rope(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                  CudaExecutionView execution);

/**
 * Direct-gain per-head RMSNorm followed by full-head split-half 2-D RoPE:
 *   n[d] = x[d]*weight[d]/sqrt(mean(x^2)+epsilon)
 *   phi(i,t) = positions[t + (i/16)*N] * theta^(-2*(i%16)/32), i=0..31
 *   out[i] = n[i]*cos(phi) - n[i+32]*sin(phi)
 *   out[i+32] = n[i+32]*cos(phi) + n[i]*sin(phi).
 *
 * q/k are contiguous BF16 [64,16,N], N>=0, and are completely overwritten in place.
 * Weights are contiguous BF16 [64] direct gains; positions is contiguous Device I32
 * [N,2], with all height coordinates followed by all width coordinates. Epsilon and
 * theta are positive and finite. Normalization and rotation use FP32 with no BF16
 * intermediate between them; the final result is BF16.
 *
 * q/k are 4-byte aligned, disjoint from each other and all read-only inputs. Weights
 * may share backing; weights/positions are naturally aligned and preserved. All
 * operands belong to execution.device(). N=0 performs no device work. No workspace,
 * persistent state or hidden allocation. Graph replay consumes current input values.
 */
void rmsnorm_rope_2d(const Tensor& positions, const Tensor& q_weight, const Tensor& k_weight,
                     float epsilon, float theta, Tensor& q, Tensor& k, CudaExecutionView execution);

} // namespace ninfer::ops
