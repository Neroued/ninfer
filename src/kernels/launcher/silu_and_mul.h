#pragma once

// qus::kernels::detail — private launch prototype for silu_and_mul. Included by the wrapper
// (host) and defined by the launcher (.cu). Not part of the public api.
// See docs/l1-kernel-layering.md §4.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

// Host entry; assumes inputs already validated by the wrapper.
void silu_and_mul_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
