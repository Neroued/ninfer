#pragma once

// qus::kernels - depthwise causal conv1d over [C,T], kernel width 4, fused SiLU.
// x/out: [C,T], weight: [C,4], conv_state: [C,3] in-out. All tensors are BF16.

#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

// Prefill with distinct read/write conv state. The initial width-3 window is
// read from `conv_state_in` and the running window is written to `conv_state_out`.
// Passing the same tensor for both is the in-place form used by reset prefill and
// continuation chunks; distinct views let prefix-append prefill read a committed
// GDN snapshot slot and publish the running state to slot 0.
void causal_conv1d_prefill(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                           Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_prefill(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                           cudaStream_t stream);
void causal_conv1d_decode(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                          cudaStream_t stream);
void causal_conv1d_sequence_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                     const Tensor& initial_slot, Tensor& out,
                                     cudaStream_t stream);

} // namespace qus::kernels
