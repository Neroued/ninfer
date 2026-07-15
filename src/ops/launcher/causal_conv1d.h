#pragma once

// ninfer::ops::detail - private launch prototypes for causal_conv1d.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kCausalConvSequenceMaxTokens = 64;
inline constexpr std::int32_t kCausalConvParallelMaxTokens = 16;

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream);
void causal_conv1d_sequence_launch(const Tensor& x, const Tensor& weight,
                                   const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                   cudaStream_t stream);
void causal_conv1d_smallt_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_sequence_snapshot_launch(const Tensor& x, const Tensor& weight,
                                            Tensor& conv_states, const Tensor& initial_slot,
                                            Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
