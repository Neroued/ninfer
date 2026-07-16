#pragma once

#include "core/tensor.h"
#include "ops/linear/q5/q5_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q5_rowsplit_gemv_r16_s2_x_launch(const Tensor& x, const Weight& w, Tensor& out,
                                      cudaStream_t stream);
void q5_rowsplit_simt_r8_c4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream);
void q5_rowsplit_simt_r8_c8_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream);
void q5_rowsplit_simt_split2_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q5_rowsplit_simt_split4_exact_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void q5_rowsplit_mma_r64_c64_launch(Q5KernelVariant variant, const Tensor& x, const Weight& w,
                                    Tensor& out, cudaStream_t stream);
void q5_rowsplit_mma_r64_c128_launch(Q5KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
