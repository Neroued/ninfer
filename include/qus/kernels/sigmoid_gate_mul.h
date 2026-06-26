#pragma once

// qus::kernels - sigmoid_gate_mul: x *= sigmoid(gate), elementwise in place.
// Reference example for the L1 api/wrapper/launcher/kernel layout; see
// docs/l1-kernel-layering.md and docs/l1-operator-catalog.md section 3.4.

#include "qus/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace qus::kernels {

// x *= sigmoid(gate), elementwise in place. gate / x: identical shape, BF16, contiguous.
void sigmoid_gate_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

} // namespace qus::kernels
