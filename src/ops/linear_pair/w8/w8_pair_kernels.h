#pragma once

#include "core/tensor.h"

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
void w8_pair_simt_r8_c4_launch(bool full, const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream);
void w8_pair_gemm_mma_r32_c64_launch(bool full, const Tensor& x, const Weight& first_weight,
                                     const Weight& second_weight, Tensor& first_out,
                                     Tensor& second_out, cudaStream_t stream);
void w8_pair_gemm_mma_r32_c128_launch(bool full, const Tensor& x, const Weight& first_weight,
                                      const Weight& second_weight, Tensor& first_out,
                                      Tensor& second_out, cudaStream_t stream);
void w8_pair_concat_mma_launch(W8PairScheduleId schedule, bool full, const Tensor& x,
                               const Weight& first_weight, const Weight& second_weight,
                               Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops::detail
