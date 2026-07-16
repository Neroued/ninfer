#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Maximum transient capacity required by any admitted route in [1,max_tokens].
[[nodiscard]] std::size_t gdn_gating_proj_workspace_bytes(std::int32_t max_tokens);

/**
 * Fuses two BF16 projections with Gated DeltaNet gate preparation. For each h,t:
 *
 *   a[h,t]    = BF16(sum_k float(a_weight[h,k]) * float(x[k,t]))
 *   b[h,t]    = BF16(sum_k float(b_weight[h,k]) * float(x[k,t]))
 *   g[h,t]    = -exp(A_log[h]) * softplus(float(a[h,t]) + dt_bias[h])
 *   beta[h,t] = sigmoid(float(b[h,t])).
 *
 * `x` is contiguous BF16 [5120,T]; both weights are contiguous BF16_CTRL [48,5120]; A_log and
 * dt_bias are contiguous FP32 [48]; g and beta are distinct contiguous FP32 [48,T]. The explicit
 * BF16 projection round is part of the numeric contract. All inputs and outputs are
 * non-overlapping. `ws` provides the transient capacity reported above and is scoped to the call;
 * there is no persistent state side effect.
 */
void gdn_gating_proj(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                     const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                     Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops
