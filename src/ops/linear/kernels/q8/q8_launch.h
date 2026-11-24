#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_a16_gemv_r4_w1_k16384(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r8_t4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_simt_r8_t8(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r32_t128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r48_t64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64_t96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64_t128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r96_t96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r128_t64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r128_t80(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x16_t48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_mma_r64x32_t64_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q8_a16_simt_r4_t4_w2_g16_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t16_w8_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t16_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_q8_a16_sliced_r16_t32_w4_s2(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
