#pragma once

// qus::kernels — silu_mul (SwiGLU activation): out = silu(gate) * up, elementwise.
// Reference example for the L1 api/wrapper/launcher/kernel layout; see
// docs/kernel-development.md and docs/design.md §5.

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

// out = silu(gate) * up, elementwise. gate / up / out: identical shape, BF16, contiguous.
void silu_mul(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels
