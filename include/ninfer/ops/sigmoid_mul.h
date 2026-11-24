#pragma once

#include "core/device.h"
#include "core/tensor.h"


namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   ideal[i] = x[i] * (1 / (1 + exp(-gate[i]))).
 *
 * gate/x are same-shaped, contiguous, non-overlapping BF16 tensors.
 * Updates x in place and preserves gate. Empty tensors perform no device work.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, CudaExecutionView execution);

} // namespace ninfer::ops
