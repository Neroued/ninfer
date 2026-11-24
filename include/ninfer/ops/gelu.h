#pragma once

#include "core/device.h"
#include "core/tensor.h"


namespace ninfer::ops {

enum class GeluMode {
    Exact,
    Tanh,
};

/**
 * Applies GELU elementwise in place. For z=x[i]:
 *
 *   Exact: ideal[i] = 0.5*z*(1 + erf(z/sqrt(2)))
 *   Tanh:  ideal[i] = 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3))).
 *
 * x is a contiguous BF16 tensor. Updates only x; empty tensors perform no device work.
 */
void gelu(Tensor& x, GeluMode mode, CudaExecutionView execution);

} // namespace ninfer::ops
