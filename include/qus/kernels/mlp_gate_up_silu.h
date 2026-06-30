#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void mlp_gate_up_silu_decode(const Tensor& x, const Weight& gate_up_weight, Tensor& out,
                             cudaStream_t stream);

} // namespace qus::kernels
