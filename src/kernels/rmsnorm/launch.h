#pragma once

// ninfer::kernels::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels::detail
