#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * x[d,r] += bias[d], where r flattens all axes above the fastest dimension D.
 *
 * Bias is contiguous BF16 [D]; x is contiguous BF16 with ne[0]=D. Adds the represented
 * values and stores BF16 in place. Bias and x must not overlap. Bias is preserved;
 * an empty x does no work. No workspace or other state is used.
 */
void add_bias(const Tensor& bias, Tensor& x, CudaExecutionView execution);

} // namespace ninfer::ops
