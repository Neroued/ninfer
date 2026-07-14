#pragma once

// ninfer::kernels - residual_add: x += y, elementwise in place.
// Reference example for the L1 api/wrapper/launcher/kernel layout; see
// docs/kernel-development.md and docs/design.md section 5.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::kernels {

// x += y, elementwise in place. y / x: identical shape, BF16, contiguous.
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);

} // namespace ninfer::kernels
