#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

enum class GeluMode {
    Exact,
    Tanh,
};

/**
 * Applies GELU elementwise in place. For z=x[i]:
 *
 *   Exact: x'[i] = BF16(0.5*z*(1 + erf(z/sqrt(2))))
 *   Tanh:  x'[i] = BF16(0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3)))).
 *
 * `x` is an arbitrary-rank contiguous BF16 tensor. The oracle evaluates the selected expression in
 * FP64 before converting the observable result to BF16. Private kernel arithmetic is
 * implementation-defined. The Op mutates only x and uses no workspace or persistent state.
 */
void gelu(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace ninfer::ops
