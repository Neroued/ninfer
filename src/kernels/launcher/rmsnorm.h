#pragma once

// qus::kernels::detail - private launch prototype for rmsnorm.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
