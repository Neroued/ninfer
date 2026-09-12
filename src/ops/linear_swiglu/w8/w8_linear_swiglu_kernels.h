#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_linear_swiglu_decode_pair_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);
void w8_linear_swiglu_splitk_exact_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                            cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c32_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c48_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r32_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c96_launch(const Tensor& x, const Weight& w, Tensor& out,
                                         cudaStream_t stream);
void w8_linear_swiglu_mma_r64_c128_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void w8_linear_swiglu_mma_r128_c64_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void w8_linear_swiglu_mma_r128_c80_launch(const Tensor& x, const Weight& w, Tensor& out,
                                          cudaStream_t stream);
void w8_dflash2_linear_swiglu_small_t_launch(const Tensor& x, const Weight& w, Tensor& out,
                                             cudaStream_t stream);

void w8_dflash2_linear_swiglu_mma_r32_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void w8_dflash2_linear_swiglu_mma_r64_c64_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void w8_dflash2_linear_swiglu_mma_r64_c80_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

void w8_dflash2_linear_swiglu_mma_r64_c96_k128_launch(const Tensor&, const Weight&, Tensor&,
                                                      cudaStream_t);

} // namespace ninfer::ops::detail
