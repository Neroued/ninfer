#pragma once

// qus::kernels::detail - private launch prototype for rope. Included by the wrapper
// and defined by the CUDA launcher.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void rope_launch(const Tensor& positions, int rotary_dim, float theta, Tensor& q, Tensor& k,
                 cudaStream_t stream);

} // namespace qus::kernels::detail
