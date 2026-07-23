#pragma once

#include "core/tensor.h"
#include "ops/linear/w8/w8_rowsplit_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

enum class W8PairScheduleId;

void w8_pair_decode_r4_launch(const Tensor& x, const Weight& first_weight,
                              const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                              cudaStream_t stream);
void w8_pair_decode_r8_launch(const Tensor& x, const Weight& first_weight,
                              const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                              cudaStream_t stream);
void w8_pair_decode_r16_launch(const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream);
void w8_pair_splitk_exact_t_launch(const Tensor& x, const Weight& first_weight,
                                   const Weight& second_weight, Tensor& first_out,
                                   Tensor& second_out, cudaStream_t stream);
void w8_pair_splitk_medium_launch(W8PairScheduleId schedule, const Tensor& x,
                                  const Weight& first_weight, const Weight& second_weight,
                                  Tensor& first_out, Tensor& second_out, cudaStream_t stream);
void w8_pair_gemm_mma_r32_c64_launch(W8KernelVariant variant, const Tensor& x,
                                     const Weight& first_weight, const Weight& second_weight,
                                     Tensor& first_out, Tensor& second_out, cudaStream_t stream);
void w8_pair_gemm_mma_r32_c80_launch(W8KernelVariant variant, const Tensor& x,
                                     const Weight& first_weight, const Weight& second_weight,
                                     Tensor& first_out, Tensor& second_out, cudaStream_t stream);
void w8_pair_gemm_mma_r32_c96_launch(W8KernelVariant variant, const Tensor& x,
                                     const Weight& first_weight, const Weight& second_weight,
                                     Tensor& first_out, Tensor& second_out, cudaStream_t stream);
void w8_pair_gemm_mma_r32_c112_launch(W8KernelVariant variant, const Tensor& x,
                                      const Weight& first_weight, const Weight& second_weight,
                                      Tensor& first_out, Tensor& second_out, cudaStream_t stream);
void w8_pair_gemm_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& first_weight,
                             const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                             cudaStream_t stream);
void w8_pair_concat_mma_launch(W8PairScheduleId schedule, W8KernelVariant variant, const Tensor& x,
                               const Weight& first_weight, const Weight& second_weight,
                               Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
