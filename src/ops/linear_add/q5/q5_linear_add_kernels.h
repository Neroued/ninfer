#pragma once

#include "core/tensor.h"
#include "ops/linear/q5/q5_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);
void q5_linear_add_mma_r64_c64_launch(Q5KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void q5_linear_add_mma_r64_c128_launch(Q5KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
