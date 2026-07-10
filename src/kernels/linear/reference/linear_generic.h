#pragma once

#include "kernels/linear/plan/linear_plan.h" // LinearFormat
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels::detail {

// Low-bit (Q4/Q5/Q6/W8G32): launcher selects the codec by fmt. w carries
// payload/qdata/qhigh + padded_shape. The small-T streaming GEMM is the universal
// low-bit path outside the tuned T==1 GEMVs and the LargeT tensor-core GEMM.
void linear_rowsplit_gemm_smallt_launch(const Tensor& x, const Weight& w, Tensor& out,
                                        LinearFormat fmt, cudaStream_t stream);
// LargeT (T > tau) tensor-core path: bf16 mma.sync with on-chip low-bit dequant.
void linear_rowsplit_gemm_mma_launch(const Tensor& x, const Weight& w, Tensor& out,
                                     LinearFormat fmt, cudaStream_t stream);
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

} // namespace qus::kernels::detail
