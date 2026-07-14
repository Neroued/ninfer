#pragma once

// ninfer::kernels::detail - private launch prototype for l2norm.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void l2norm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels::detail
