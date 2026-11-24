#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * out[d,r] = x[d,r] / sqrt(sum_j x[j,r]^2 + eps).
 *
 * Reduces the fastest dimension D=ne[0]; r flattens the remaining axes. Input and output are
 * same-shaped contiguous BF16 tensors with disjoint storage. Eps is positive and finite.
 * Writes all of out and preserves x. An empty tensor does no work. No workspace or state is used.
 */
void l2norm(const Tensor& x, float eps, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
