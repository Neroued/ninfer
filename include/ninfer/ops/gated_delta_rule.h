#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient arena capacity required by gated_delta_rule for the given geometry. It is
 * zero when the private implementation requires no transient storage. `head_dim` must be one of
 * {16,32,64,128}; `value_heads` must be at least `qk_heads` and divisible by it.
 */
[[nodiscard]] std::size_t gated_delta_rule_workspace_bytes(std::int32_t head_dim,
                                                           std::int32_t qk_heads,
                                                           std::int32_t value_heads,
                                                           std::int32_t tokens);

/**
 * Applies the Gated DeltaNet recurrence independently for each value head h. Let
 * G=value_heads/qk_heads; its Q/K head is qh=floor(h/G). Starting from S_h, for t in increasing
 * order:
 *
 *   alpha       = exp(g[h,t])
 *   delta       = beta[h,t] * (v[:,h,t] - alpha * S_h * k[:,qh,t])
 *   S_h         = alpha * S_h + outer(delta, k[:,qh,t])
 *   out[:,h,t]  = BF16(scale * S_h * q[:,qh,t]).
 *
 * Shapes/dtypes are contiguous q/k BF16 [S,Hqk,T], v/out BF16 [S,Hv,T], g/beta FP32 [Hv,T], and
 * state FP32 [S,S,Hv], where S is one of {16,32,64,128}, Hqk>=1, Hv>=Hqk, and Hv%Hqk==0. `scale`
 * is 1/sqrt(S). Inputs and out do not overlap state or one another. `ws` supplies transient
 * storage reported by gated_delta_rule_workspace_bytes; scratch is scoped to the call. T may be
 * any positive value.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, WorkspaceArena& ws, Tensor& ssm_state,
                      Tensor& out, cudaStream_t stream);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` may be disjoint or exactly the same storage. No other
 * arguments may overlap either state.
 */
void gated_delta_rule(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, WorkspaceArena& ws,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream);

/**
 * Snapshot form of the same recurrence. `ssm_states` is contiguous FP32 [S,S,Hv,Slots],
 * `initial_slot` is a contiguous I32 scalar in [0,Slots), and Slots>=T. The selected slot supplies
 * the initial state; after token t, the new state is written to slot t. Slots at and above T are
 * unchanged. This form uses no arena allocation, and `ssm_states` is the only persistent state
 * mutated.
 */
void gated_delta_rule_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, WorkspaceArena& ws,
                               Tensor& ssm_states, const Tensor& initial_slot, Tensor& out,
                               cudaStream_t stream);

} // namespace ninfer::ops
