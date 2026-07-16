#pragma once

#include "core/tensor.h"
#include "ops/linear/q4/q4_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_linear_swiglu_gemv_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                       cudaStream_t stream);
void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(Q4KernelVariant variant, const Tensor& x,
                                                          const Weight& w, Tensor& out,
                                                          cudaStream_t stream);

} // namespace ninfer::ops::detail
