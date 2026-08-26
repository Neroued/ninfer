#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient arena capacity required by gated_delta_net for the given geometry. It is
 * zero when the private implementation requires no transient storage. The state/head dimension is
 * fixed at 128; `value_heads` must be at least `qk_heads` and divisible by it. The query covers
 * every T in the inclusive interval and throws for an invalid profile or interval.
 */
[[nodiscard]] std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                                   std::int32_t value_heads,
                                                                   bool normalize_qk,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

/**
 * Applies the Gated DeltaNet recurrence independently for each value head h. Let
 * G=value_heads/qk_heads; its Q/K head is qh=floor(h/G). Starting from S_h, for t in increasing
 * order:
 *
 *   alpha       = exp(g[h,t])
 *   delta       = beta[h,t] * (v[:,h,t] - alpha * S_h * k[:,qh,t])
 *   S_h         = alpha * S_h + outer(delta, k[:,qh,t])
 *   ideal[:,h,t] = scale * S_h * q[:,qh,t].
 *
 * Shapes/dtypes are contiguous q/k BF16 [128,Hqk,T], v/out BF16 [128,Hv,T], g/beta FP32 [Hv,T],
 * and state FP32 [128,128,Hv], where Hqk>=1, Hv>=Hqk, and Hv%Hqk==0. `scale` is 1/sqrt(128). When
 * `normalize_qk` is true, the recurrent implementation consumes raw q/k and applies
 * x / sqrt(sum(x^2) + 1e-6) independently to every 128-element row before using it. When false,
 * q/k are consumed as supplied. The oracle evaluates the complete recurrence and `ideal` naively
 * in FP64 from the represented inputs and FP32 initial state. The BF16 out is promoted and
 * compared directly with that result; output storage rounding belongs to the Op's numerical
 * criterion, not the oracle. Recurrent implementations may apply the normalization directly;
 * chunked implementations may use private normalized staging. The corresponding private storage
 * is included by gated_delta_net_workspace_capacity_bytes when `normalize_qk` is true.
 * Inputs and out do not overlap state or one another. `ws` supplies transient storage reported by
 * gated_delta_net_workspace_capacity_bytes; scratch is scoped to the call. T may be any positive
 * value.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     Tensor& ssm_state, Tensor& out, cudaStream_t stream);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` may be disjoint or exactly the same storage. No other
 * arguments may overlap either state.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     cudaStream_t stream);

/**
 * One-token update for B independent state-pool slots. q/k are contiguous BF16 [128,Hqk,1,B],
 * v/out are BF16 [128,Hv,1,B], g/beta are FP32 [Hv,1,B], `ssm_states` is contiguous FP32
 * [128,128,Hv,Slots], and both selector tensors are contiguous device I32 [B]. B is in [1,8].
 * Row b reads source_state_slots[b] and publishes the transition to
 * destination_state_slots[b]; selectors may be equal for an in-place row. Destination slots are
 * distinct across active rows. This form uses no arena allocation.
 */
void gated_delta_net_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  bool normalize_qk, Tensor& ssm_states,
                                  const Tensor& source_state_slots,
                                  const Tensor& destination_state_slots, Tensor& out,
                                  cudaStream_t stream);

/**
 * Op: gated_delta_net_replay_record
 *
 * Evaluates B independent normalized Gated DeltaNet recurrences from absolute state-pool slots
 * without modifying any state. q/k are BF16 [128,Hq,T,B], v/out are BF16 [128,Hv,T,B], g/beta
 * are FP32 [Hv,T,B], and ssm_states is FP32 [128,128,Hv,S]. The ReplaySSM execution domain is
 * B=1..8 and T=2..16, with Hq=16 and Hv in {32,48}. scale is 1/sqrt(128).
 *
 * valid_columns is empty for dense rows or device I32 [B], with every caller-supplied extent in
 * [1,T]. initial_state_slots is device I32 [B] containing absolute slots in [0,S). For each valid
 * transition, key_record BF16 [128,Hq,T,B], value_record BF16 [128,Hv,T,B], and gate_record FP32
 * [2,Hv,T,B] receive bit-preserving copies of raw k, v, and {g,beta}. The invalid record suffix is
 * unchanged, while the invalid out suffix is exact BF16 zero. Inputs, state, records, and out are
 * pairwise non-overlapping.
 */
void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninfer::ops
