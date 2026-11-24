#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * mean[r] = sum_d x[d,r] / D; variance[r] = sum_d (x[d,r]-mean[r])^2 / D.
 * out[d,r] = (x[d,r]-mean[r]) / sqrt(variance[r]+eps) * weight[d] + bias[d].
 *
 * Reduces the fastest dimension D=ne[0]; r flattens the remaining axes. Input and output are
 * same-shaped contiguous BF16 tensors; weight and bias are contiguous BF16 [D]. All tensors
 * have disjoint storage and eps is positive and finite. Writes all of out and preserves inputs.
 * An empty tensor does no work. No workspace or state is used.
 */
void layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps, Tensor& out,
                CudaExecutionView execution);

} // namespace ninfer::ops
