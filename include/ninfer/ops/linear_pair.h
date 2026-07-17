#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Computes two same-shaped projections of one input:
 *
 *   first_out[:,t]  = linear(x[:,t], first_weight)
 *   second_out[:,t] = linear(x[:,t], second_weight).
 *
 * `x` is contiguous BF16 [K,T], both outputs are distinct contiguous BF16 [N,T], and both
 * weights have the same logical [N,K] shape and RowSplit W8G32_F16S encoding with FP16 scales.
 * The current registry admits exactly the paired [1024,5120] physical projection for every
 * positive T. Numeric semantics are those of linear(). Inputs, outputs, and weight planes must be
 * mutually non-overlapping. `ws` is
 * caller-owned transient scratch; the Op has no persistent state side effect.
 */
void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
