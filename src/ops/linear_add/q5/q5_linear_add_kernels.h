#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q5_linear_add_ksplit_mma_residual_launch(const Tensor& x, const Weight& w,
                                              Tensor& residual_out, cudaStream_t stream);
void q5_linear_add_mma_r64_c32_s3_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream);
void q5_linear_add_mma_r64_c32_s4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                         cudaStream_t stream);
void q5_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);

} // namespace ninfer::ops::detail
