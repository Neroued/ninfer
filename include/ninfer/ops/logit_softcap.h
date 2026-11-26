#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * out[i] = softcap * tanh(FP32(x[i]) / softcap), rounded to the operand dtype.
 * Softcap is positive and finite. X and out have matching shapes and are
 * contiguous FP32 or BF16. Nonempty operands are naturally aligned and on the
 * execution device. Exact in-place alias is allowed, and partial overlap is forbidden.
 *
 * Empty tensors perform no device work. No workspace or state. Graph replay reads
 * the current values of x.
 */
void logit_softcap(const Tensor& x, float softcap, Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
