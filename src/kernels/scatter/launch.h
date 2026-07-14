#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void scatter_launch(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream);

} // namespace ninfer::kernels::detail
