#pragma once

// qus::kernels - depthwise causal conv1d over [C,T], kernel width 4, fused SiLU.
// x/out: [C,T], weight: [C,4], conv_state: [C,3] in-out. All tensors are BF16.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

void causal_conv1d_prefill(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                           cudaStream_t stream);
void causal_conv1d_decode(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                          cudaStream_t stream);
void causal_conv1d_sequence_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                     const Tensor& initial_slot, Tensor& out,
                                     cudaStream_t stream);

} // namespace qus::kernels
