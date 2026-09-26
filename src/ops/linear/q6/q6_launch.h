#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
using Q6Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q6_a16_simt_r8_t4(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream);
void launch_q6_a16_gemv_r4_w2_g16(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream);
void launch_q6_a16_sliced_r16_t8_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void launch_q6_a16_sliced_r32_t16_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r32_t32_w4_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r32_t64_w2_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r16_t24_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_sliced_r32_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q6_a16_mma_r64_t40_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t48_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t56_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t64_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t72_k128(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q6_a16_mma_r64_t80(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);
void launch_q6_a16_mma_r64_t96(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);
void launch_q6_a16_mma_r64_t128(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
void launch_q6_a16_mma_r64_t112(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
} // namespace ninfer::ops::detail
