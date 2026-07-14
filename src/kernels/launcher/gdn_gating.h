#pragma once

// ninfer::kernels::detail - private launch prototype for gdn_gating.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& A_log,
                       const Tensor& dt_bias, Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::kernels::detail
