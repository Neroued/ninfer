#pragma once

// qus::kernels::detail - private launch prototypes for linear variants.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

void linear_dense_gemv_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                              cudaStream_t stream);
void linear_dense_gemm_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                              cudaStream_t stream);
void linear_q4_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void linear_q4_gemm_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void linear_q5_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void linear_q5_gemm_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void linear_q6_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void linear_q6_gemm_launch(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace qus::kernels::detail
