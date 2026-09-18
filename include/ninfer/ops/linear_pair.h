#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Validates the native weight pair and inclusive T interval. Existing routes use no scratch.
[[nodiscard]] std::size_t linear_pair_workspace_capacity_bytes(const Weight& first_weight,
                                                               const Weight& second_weight,
                                                               std::int32_t min_tokens,
                                                               std::int32_t max_tokens);

/**
 * Computes two same-shaped projections of one input:
 *
 *   first_out[:,t]  = linear(x[:,t], first_weight)
 *   second_out[:,t] = linear(x[:,t], second_weight).
 *
 * `x` is contiguous BF16 [K,T], both outputs are distinct contiguous BF16 [N,T], and both
 * weights have the same logical [N,K] shape and RowSplit Q8_G32_FP16 encoding with FP16 scales.
 * The registry admits the paired [1024,5120] physical projection and the exact adjacent
 * [1024,2048] K/V row views (parent rows [4096,5120) and [5120,6144)) for every positive T.
 * Numeric semantics are those of linear(). Inputs, outputs, and weight planes must be mutually
 * non-overlapping. The Op requires no caller-owned transient storage and has no persistent state
 * side effect.
 */
void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops
