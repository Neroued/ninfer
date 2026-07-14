#pragma once

// ninfer::kernels - depthwise causal conv1d over [C,T], kernel width 4, fused SiLU.
// x/out: [C,T], weight: [C,4], conv_state: [C,3] in-out. All tensors are BF16.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

// In-place state form. T==1 uses the decode launcher; larger T uses the prefill launcher.
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        cudaStream_t stream);

// Distinct state form used by prefix-append prefill. The initial width-3 window is read from
// conv_state_in and the final window is written to conv_state_out.
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out, cudaStream_t stream);

// Snapshot-state semantics used by speculative verification.
void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                 const Tensor& initial_slot, Tensor& out, cudaStream_t stream);

} // namespace ninfer::kernels
