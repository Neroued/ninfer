#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Applies RMS normalization over ne[0] and an elementwise SiLU gate. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   out[d,r] = BF16(x[d,r] * inv_r * weight[d] * SiLU(z[d,r])).
 *
 * `x`, `z`, and `out` are same-shaped contiguous BF16 tensors, `weight` is contiguous BF16 [D],
 * and eps is positive and finite. This form does not apply a unit offset to weight. Inputs and
 * output must not overlap. The oracle evaluates the complete expression naively in FP64 before
 * converting the observable output to BF16. Kernel reduction, staging, and accumulator precision
 * are implementation choices. There is no workspace or persistent state side effect.
 */
void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream);

} // namespace ninfer::ops
