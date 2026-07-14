#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

// Column overwrite: dst[:, indices[i]] = src[:, i]. The model layer owns index validation.
void scatter(const Tensor& src, const Tensor& indices, Tensor& dst, cudaStream_t stream);

} // namespace ninfer::kernels
