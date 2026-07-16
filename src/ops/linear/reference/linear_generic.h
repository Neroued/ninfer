#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Dense (BF16/FP32): wrapper passes as_dense(w) as the weight Tensor.
void linear_generic_dense_gemv_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream);
void linear_generic_dense_gemm_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
