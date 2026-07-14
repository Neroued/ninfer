#pragma once

// ninfer::kernels::detail - private launch prototypes for causal_conv1d.

#include "ninfer/core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels::detail {

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream);
void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, Tensor& conv_state,
                                 Tensor& out, cudaStream_t stream);
void causal_conv1d_sequence_snapshot_launch(const Tensor& x, const Tensor& weight,
                                            Tensor& conv_states, const Tensor& initial_slot,
                                            Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels::detail
