#pragma once
#include "core/weight.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
void q5_linear_add_split2_exact_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t8_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t16_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r16_t24_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w4_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t24_pairwise_launch(const Tensor&, const Weight&, Tensor&,
                                                  cudaStream_t);
void q5_linear_add_sliced_r32_t32_w4_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t32_w2_s2_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_sliced_r32_t64_w2_s1_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r32_t32_k128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r32_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
void q5_linear_add_mma_r64_t128_launch(const Tensor&, const Weight&, Tensor&, cudaStream_t);
} // namespace ninfer::ops::detail
