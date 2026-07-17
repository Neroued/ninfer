#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Depthwise causal width-4 convolution followed by SiLU. Let u[c,-3..-1] be the three values in
 * the input state and u[c,t]=x[c,t] for t>=0. Then
 *
 *   out[c,t] = BF16(SiLU(sum_{j=0..3} weight[c,j] * u[c,t-3+j])).
 *
 * `x` and `out` are contiguous BF16 [C,T], `weight` is contiguous BF16 [C,4], and a state is
 * contiguous BF16 [C,3] ordered oldest to newest. The oracle evaluates the convolution and SiLU
 * naively in FP64 before converting the observable output to BF16. Kernel accumulator and staging
 * precision are implementation choices. Input, weight, output, and state storage do not overlap
 * except for the explicitly allowed exact alias between state input and state output. No caller
 * workspace is used. T may be any positive value.
 */

// Reads conv_state as the initial window and replaces it with the final three values after x.
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        cudaStream_t stream);

// Distinct-state form. conv_state_in and conv_state_out may be disjoint or exactly the same
// storage; conv_state_out receives the trailing width-3 window of concat(conv_state_in,x).
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out, cudaStream_t stream);

/**
 * Snapshot form. `conv_states` is contiguous BF16 [C,3,Slots], `initial_slot` is a contiguous I32
 * scalar selecting an initial window in [0,Slots), and Slots>=T. After token t, the resulting
 * [C,3] window is written to snapshot slot t; slots at and above T are unchanged. `conv_states`
 * is the only persistent state mutated. It must not overlap x, weight, initial_slot, or out.
 */
void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                 const Tensor& initial_slot, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
