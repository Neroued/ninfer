#pragma once

#include "core/tensor.h"
#include "ops/linear/q4/q4_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_rowsplit_gemv_r4_w1_shared_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q4_rowsplit_gemv_r4_w1_direct_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q4_rowsplit_gemv_r1_w8_direct_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);

void q4_rowsplit_gemm_simt_r8_c4_launch(Q4KernelVariant variant, const Tensor& x, const Weight& w,
                                        Tensor& out, cudaStream_t stream);
void q4_rowsplit_gemm_simt_r8_c8_launch(Q4KernelVariant variant, const Tensor& x, const Weight& w,
                                        Tensor& out, cudaStream_t stream);

void q4_rowsplit_gemm_mma_r64_c64_launch(Q4KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void q4_rowsplit_gemm_mma_r64_c128_launch(Q4KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
