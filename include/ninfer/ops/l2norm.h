#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Normalizes each logical row over the fastest dimension D=ne[0]:
 *
 *   inv_r    = 1 / sqrt(sum_{d=0..D-1} x[d,r]^2 + eps)
 *   out[d,r] = BF16(x[d,r] * inv_r).
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors and eps is positive and finite. Input
 * and output must not overlap. The oracle evaluates reduction and scaling naively in FP64 before
 * converting the observable output to BF16. Kernel reduction, staging, and accumulator precision
 * are implementation choices. There is no workspace or persistent state side effect.
 */
void l2norm(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
