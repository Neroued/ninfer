#pragma once

// ninfer::kernels::detail - private launch prototype for sigmoid_mul. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/kernel-development.md §2.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

// Host entry; assumes inputs already validated by the wrapper.
void sigmoid_gate_mul_launch(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace ninfer::kernels::detail
