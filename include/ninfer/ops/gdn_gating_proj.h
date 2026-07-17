#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Maximum transient capacity required for any positive T <= max_tokens. This query does not cap T.
[[nodiscard]] std::size_t gdn_gating_proj_workspace_bytes(std::int32_t max_tokens);

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
 * from the represented inputs. Projection
 * staging, accumulator precision, and any private materialization are implementation choices and
 * are compared with that oracle under the route's FP32-output criterion. All inputs and outputs are
 * non-overlapping. `ws` provides the transient capacity reported above and is scoped to the call;
 * there is no persistent state side effect.
 */
void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream);

/**
 * Qwen3.6-35B-A3B exact storage domain. `ab_weight` is one contiguous BF16_CTRL [64,2048]
 * parent whose rows [0,32) and [32,64) are A and B. The implementation consumes those halves
 * as zero-copy views and produces FP32 g/beta [32,T] under the same logical formula and oracle.
 * All other effects and non-overlap requirements match the two-weight form.
 */
void gdn_gating_proj(const Tensor& x, const Weight& ab_weight, const Tensor& A_log,
                     const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g, Tensor& beta,
                     cudaStream_t stream);

} // namespace ninfer::ops
