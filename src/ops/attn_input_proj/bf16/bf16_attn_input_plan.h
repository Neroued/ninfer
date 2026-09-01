#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The Attention epilogue has its own measured crossover. The wider candidate domain remains
// benchmark-callable so the production boundary is not conflated with template availability.
inline constexpr std::int32_t kBf16AttnInputSmallTMinTokens   = 2;
inline constexpr std::int32_t kBf16AttnInputSmallTMaxTokens   = 32;
inline constexpr std::int32_t kBf16AttnInputSmallTDispatchEnd = 22;

void bf16_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);
void bf16_attn_input_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                Tensor& k, Tensor& v, cudaStream_t stream);
#ifdef NINFER_VOLTA_BUILD
void bf16_attn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& q,
                                         Tensor& gate, Tensor& k, Tensor& v,
                                         cudaStream_t stream);
#endif

void bf16_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                              Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops::detail
