#pragma once

#include "core/device.h"
#include "core/tensor.h"


namespace ninfer::ops {

/**
 * x[i] = x[i] + y[i].
 *
 * x/y are same-shaped, contiguous, non-overlapping BF16 tensors.
 * Updates x in place and preserves y. Empty tensors perform no device work.
 */
void residual_add(const Tensor& y, Tensor& x, CudaExecutionView execution);

} // namespace ninfer::ops
