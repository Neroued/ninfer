#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_linear_add_decode_r4_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream);
void w8_linear_add_decode_r8_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                    cudaStream_t stream);
void w8_linear_add_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& residual_out,
                                     cudaStream_t stream);
void w8_linear_add_simt_r8_c4_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_simt_r8_c8_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_splitk_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                     Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_splitk_mma_composite_launch(const Tensor& x, const Weight& w,
                                               Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r32_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r48_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                      Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c112_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r128_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);
void w8_linear_add_mma_r128_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                       Tensor& residual_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
