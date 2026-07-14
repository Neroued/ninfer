#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

// In-place channel broadcast: x[d, ...] = bf16(float(x[d, ...]) + float(bias[d])).
void add_bias(const Tensor& bias, Tensor& x, cudaStream_t stream);

} // namespace ninfer::kernels
