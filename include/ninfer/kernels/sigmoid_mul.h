#pragma once

// ninfer::kernels - sigmoid_mul: x *= sigmoid(gate), elementwise in place.
// Reference example for the L1 api/wrapper/launcher/kernel layout; see
// See docs/kernel-development.md §3.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::kernels {

// x *= sigmoid(gate), elementwise in place. gate / x: identical shape, BF16, contiguous.
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace ninfer::kernels
