#pragma once

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void add_bias_launch(const Tensor& bias, Tensor& x, cudaStream_t stream);

} // namespace qus::kernels::detail
