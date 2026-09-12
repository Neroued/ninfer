#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using W8Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_decode_r4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_small_t(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_t_splitk(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_t_composite(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_dflash_medium(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_medium_splitk_c144(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_simt_r8_c4(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_simt_r8_c8(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_mma_r32_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c112(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r48_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c112(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r128_c64(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64x16_c48_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_mma_r64x32_c64_k128_a1(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_w8_exact_mma_r32_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r32_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r48_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r48_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r64_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r64_c128(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r96_c96(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void launch_w8_exact_mma_r128_c80(const Tensor&, const Weight&, Tensor&, cudaStream_t);

} // namespace ninfer::ops::detail
