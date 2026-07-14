#pragma once

// ninfer::kernels - fused RMSNorm and SiLU gate: out = rmsnorm(x, weight) * silu(z).

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream);

} // namespace ninfer::kernels
