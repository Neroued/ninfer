#pragma once
#include "core/tensor.h"
#include "core/weight.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
using Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q4_a16_gemv_r1_w8_direct(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);
void launch_q4_a16_gemv_r1_w8_k6144(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_a16_gemv_r4_w1_direct(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);
void launch_q4_a16_mma_r32_t128_k64_s2_a2(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
void launch_q4_a16_mma_r32_t32_k128_s2_a2(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
void launch_q4_a16_mma_r32_t32_k64_wr16_wt16_s3_a3_b2(const Tensor& x, const Weight& weight,
                                                      Tensor& out, cudaStream_t stream);
void launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s2_a2_b2(const Tensor& x, const Weight& weight,
                                                      Tensor& out, cudaStream_t stream);
void launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s3_a3_b2(const Tensor& x, const Weight& weight,
                                                      Tensor& out, cudaStream_t stream);
void launch_q4_a16_mma_r64_t112(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
void launch_q4_a16_mma_r64_t120(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
void launch_q4_a16_mma_r64_t128(const Tensor& x, const Weight& weight, Tensor& out,
                                cudaStream_t stream);
void launch_q4_a16_mma_r64_t48(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);
void launch_q4_a16_mma_r64_t64_k128_s2_a1(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
void launch_q4_a16_mma_r64_t64_k64_wr32_wt16_s2_a2_b2(const Tensor& x, const Weight& weight,
                                                      Tensor& out, cudaStream_t stream);
void launch_q4_a16_mma_r64_t72(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);
void launch_q4_a16_mma_r64_t80(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);
void launch_q4_a16_mma_r64_t96(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);
void launch_q4_a16_simt_r4_t1_w2_g8_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void launch_q4_a16_simt_r4_t4_w2_g8_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void launch_q4_a16_sliced_k2048_t4(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream);
void launch_q4_a16_sliced_k5120_t16(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_a16_sliced_k5120_t24(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);
void launch_q4_a16_sliced_k5120_t4(const Tensor& x, const Weight& weight, Tensor& out,
                                   cudaStream_t stream);
void launch_q4_a16_sliced_r16_t16_w2_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r16_t16_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r16_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r16_t8_capacity4(const Tensor& x, const Weight& weight, Tensor& out,
                                           cudaStream_t stream);
void launch_q4_a16_sliced_r16_t8_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void launch_q4_a16_sliced_r32_t16_w4_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r32_t16_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r32_t32_w2_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r32_t32_w4_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r32_t32_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r32_t64_w2_s1(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void launch_q4_a16_sliced_r32_t8_w4_s2(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
} // namespace ninfer::ops::detail
