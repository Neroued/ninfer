#pragma once

// qus::kernels - residual_add: x += y, elementwise in place.
// Reference example for the L1 api/wrapper/launcher/kernel layout; see
// docs/l1-kernel-layering.md and docs/design.md section 5.

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

// x += y, elementwise in place. y / x: identical shape, BF16, contiguous.
void residual_add(const Tensor& y, Tensor& x, cudaStream_t stream);

} // namespace qus::kernels
