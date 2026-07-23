#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_linear_swiglu_decode_pair_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_decode_pair_r4_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream);
void w8_linear_swiglu_decode_pair_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);
void w8_linear_swiglu_splitk_exact_t_launch(W8KernelVariant variant, const Tensor& x,
                                            const Weight& w, Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c32_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c48_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c96_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                         Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c128_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r128_c64_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream);
void w8_linear_swiglu_mma_r128_c80_launch(W8KernelVariant variant, const Tensor& x, const Weight& w,
                                          Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
