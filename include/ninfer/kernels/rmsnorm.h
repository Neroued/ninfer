#pragma once

// ninfer::kernels - rmsnorm over the fastest dimension ne[0].
// unit_offset=true computes x/rms(x) * (1 + weight); false uses plain weight.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::kernels {

void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream);

} // namespace ninfer::kernels
