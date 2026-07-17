#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Applies affine LayerNorm over the fastest dimension D=ne[0]. For each logical row r:
 *
 *   mean_r    = (1/D) * sum_d x[d,r]
 *   variance  = (1/D) * sum_d (x[d,r] - mean_r)^2
 *   out[d,r]  = BF16((x[d,r]-mean_r)/sqrt(variance+eps) * weight[d] + bias[d]).
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors; weight and bias are contiguous BF16 [D];
 * eps is positive and finite. All inputs and output must be non-overlapping. The oracle evaluates
 * moments and affine math naively in FP64 before converting the observable output to BF16. Kernel
 * reduction, staging, and accumulator precision are implementation choices. There is no workspace
 * or persistent state side effect.
 */
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out,
                cudaStream_t stream);

} // namespace ninfer::ops
