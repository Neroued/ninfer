#pragma once

#include "ops/linear/plan/linear_plan.h" // LinearFormat
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// W8 compatibility launcher until W8 owns a format-local backend.
void linear_rowsplit_gemm_smallt_launch(const Tensor& x, const Weight& w, Tensor& out,
                                        LinearFormat fmt, cudaStream_t stream);
// W8 tensor-core paths.
void linear_rowsplit_w8g32_gemm_mma_launch(const Tensor& x, const Weight& w, Tensor& out,
                                           cudaStream_t stream);
void linear_rowsplit_w8g32_kv_gemm_mma_launch(const Tensor& x, const Weight& k_weight,
                                              const Weight& v_weight, Tensor& k_out, Tensor& v_out,
                                              cudaStream_t stream);

// Dense (BF16/FP32): wrapper passes as_dense(w) as the weight Tensor.
void linear_generic_dense_gemv_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream);
void linear_generic_dense_gemm_launch(const Tensor& x, const Tensor& weight, Tensor& out,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
