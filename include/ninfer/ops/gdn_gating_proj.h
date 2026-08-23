#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// The geometry is the fixed implementation profile. Each query covers every T in the inclusive
// interval; invalid profiles or intervals throw.
[[nodiscard]] std::size_t gdn_gating_proj_workspace_capacity_bytes(std::int32_t heads,
                                                                   std::int32_t input_rows,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

// Transient capacity for the corresponding pre-normalized control Op. The query follows the same
// interval rules and does not make the optimized route part of the semantic API.
[[nodiscard]] std::size_t gdn_norm_gating_proj_workspace_capacity_bytes(std::int32_t heads,
                                                                        std::int32_t input_rows,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens);

/**
 * Fuses two BF16 projections with Gated DeltaNet gate preparation. For each h,t:
 *
 *   a[h,t]    = sum_k a_weight[h,k] * x[k,t]
 *   b[h,t]    = sum_k b_weight[h,k] * x[k,t]
 *   g[h,t]    = -exp(A_log[h]) * softplus(a[h,t] + dt_bias[h])
 *   beta[h,t] = sigmoid(b[h,t]).
 *
 * `x` is contiguous BF16 [5120,T]; both weights are contiguous BF16_CTRL [48,5120]; A_log and
 * dt_bias are contiguous FP32 [48]; g and beta are distinct contiguous FP32 [48,T]. The numerical
 * contract accepts every positive T. The oracle evaluates the logical formula naively in FP64
 * from the represented inputs. Projection staging, accumulator precision, and any private
 * materialization are implementation choices. The FP32 outputs are promoted and compared directly
 * with that oracle under the Op's named criterion. All inputs and outputs are non-overlapping.
 * `ws` provides the transient capacity reported above and is scoped to the call; there is no
 * persistent state side effect.
 */
void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream);

/**
 * Registered contiguous-parent storage forms of gdn_gating_proj:
 *
 * - Qwen3.8-27B: BF16_CTRL `ab_weight [96,5120]`, with A in rows [0,48) and B in [48,96);
 * - Qwen3.6-35B-A3B: BF16_CTRL `ab_weight [64,2048]`, with A in rows [0,32) and B in [32,64).
 *
 * The complete immutable parent is the public weight. Its halves are consumed as zero-copy views
 * and produce FP32 g/beta `[heads,T]` under the same logical formula and oracle. All other effects
 * and non-overlap requirements match the two-weight form.
 */
void gdn_gating_proj(const Tensor& x, const Weight& ab_weight, const Tensor& A_log,
                     const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g, Tensor& beta,
                     cudaStream_t stream);

/**
 * Applies the Qwen3.6 GDN input RMSNorm and control projection as one semantic Op:
 *
 *   n[k,t]    = x[k,t] * rsqrt(mean_j(x[j,t]^2) + eps) * (1 + norm_weight[k])
 *   h_ideal[k,t] = n[k,t]
 *   a[r,t]    = sum_k a_weight[r,k] * n[k,t]
 *   b[r,t]    = sum_k b_weight[r,k] * n[k,t]
 *   g[r,t]    = -exp(A_log[r]) * softplus(a[r,t] + dt_bias[r])
 *   beta[r,t] = sigmoid(b[r,t]).
 *
 * `h` is the explicit BF16 output consumed by the other GDN projections. Its BF16 values are
 * promoted and compared directly with `h_ideal`; final storage rounding is not reproduced inside
 * the oracle. The control branch is evaluated directly from `n` and is not required to round
 * through h. Private tensor-core operand staging remains an implementation choice. The tensor and
 * weight domains otherwise match the two-weight gdn_gating_proj form. The implementation may fuse
 * or compose its internal kernels for any positive T; that route is not observable at this
 * boundary.
 */
void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                          const Tensor& dt_bias, WorkspaceArena& ws, Tensor& h, Tensor& g,
                          Tensor& beta, cudaStream_t stream);

/** The Qwen3.8-27B and Qwen3.6-35B-A3B contiguous-parent storage forms described above. */
void gdn_norm_gating_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& ab_weight, const Tensor& A_log, const Tensor& dt_bias,
                          WorkspaceArena& ws, Tensor& h, Tensor& g, Tensor& beta,
                          cudaStream_t stream);

} // namespace ninfer::ops
