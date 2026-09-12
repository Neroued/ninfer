#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Depthwise causal width-4 convolution followed by SiLU. Let u[c,-3..-1] be the three values in
 * the input state and u[c,t]=x[c,t] for t>=0. Then
 *
 *   ideal[c,t] = SiLU(sum_{j=0..3} weight[c,j] * u[c,t-3+j]).
 *
 * `x` and `out` are contiguous BF16 [C,T], `weight` is contiguous BF16 [C,4], and a state is
 * contiguous BF16 [C,3] ordered oldest to newest. The oracle evaluates `ideal` naively in FP64 from
 * the represented inputs. The BF16 output is promoted and compared directly with that result;
 * output storage rounding belongs to the Op's numerical criterion, not the oracle. Kernel
 * accumulator and staging precision are implementation choices. Input, weight, output, and state
 * storage do not overlap except for the explicitly allowed exact alias between state input and
 * state output. No caller workspace is used. T may be any positive value.
 */

// Reads conv_state as the initial window and replaces it with the final three values after x.
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, Tensor& conv_state, Tensor& out,
                        cudaStream_t stream);

// Distinct-state form. conv_state_in and conv_state_out may be disjoint or exactly the same
// storage; conv_state_out receives the trailing width-3 window of concat(conv_state_in,x).
void causal_conv1d_silu(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                        Tensor& conv_state_out, Tensor& out, cudaStream_t stream);

// Split-output form. `ideal` is unchanged; its rows are written to three destinations instead of
// one, partitioned in row order as out0, out1, out2. Every destination is contiguous BF16 with the
// column count of x, and every operand is four-byte aligned. The supported row profiles are
// (2048, 2048, 4096) over C = 8192 and (2048, 2048, 6144) over C = 10240; any other profile is
// rejected. conv_state_in and conv_state_out follow the family rule above: disjoint or exactly the
// same storage. No destination may overlap another, x, weight, or either state.
void causal_conv1d_silu_split(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                              Tensor& conv_state_out, Tensor& out0, Tensor& out1, Tensor& out2,
                              cudaStream_t stream);

/**
 * Snapshot form for B independent sequences. `x` and `out` are contiguous BF16 [C,W,B],
 * `conv_states` is contiguous BF16 [C,3,Slots], and `initial_state_slots` and
 * `snapshot_base_slots` are contiguous I32 [B]. `valid_columns` is either contiguous I32 [B],
 * with every value in [1,W], or an empty Tensor meaning every row has W valid columns. B=1
 * accepts every positive W; B=2..8 accepts W=1..16.
 *
 * Row b starts from the window selected by initial_state_slots[b]. After valid column j, the new
 * window is written to snapshot_base_slots[b]+j. Invalid-tail output columns are exact BF16 zero
 * and do not mutate state. The caller reserves the complete [base,base+W) interval for every row;
 * all row reservations are disjoint and no row overwrites another row's initial slot. A row's own
 * initial slot may lie in its destination reservation. `conv_states` is the only persistent state
 * mutated and must not overlap x, weight, metadata, or out.
 */
void causal_conv1d_silu_snapshot(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                 const Tensor& valid_columns, const Tensor& initial_state_slots,
                                 const Tensor& snapshot_base_slots, Tensor& out,
                                 cudaStream_t stream);

} // namespace ninfer::ops
