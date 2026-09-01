#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q5_linear_add_gemv_residual_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                        cudaStream_t stream);
void q5_linear_add_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);
void q5_linear_add_simt_wide_t_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c24_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                      cudaStream_t stream);
void q5_linear_add_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                       cudaStream_t stream);

} // namespace ninfer::ops::detail
