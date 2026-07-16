#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_pair_gemm_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& first_weight,
                             const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
