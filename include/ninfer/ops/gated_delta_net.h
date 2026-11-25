#pragma once

#include "ninfer/ops/ragged.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"
#include "core/record_view.h"


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
 * and state FP32 or BF16 [128,128,Hv], where Hqk>=1, Hv>=Hqk, and Hv%Hqk==0. `scale` is
 * 1/sqrt(128). When `normalize_qk` is true, the Op consumes raw q/k and applies
 * x / sqrt(sum(x^2) + 1e-6) independently to every 128-element row before using it. When false,
 * q/k are consumed as supplied. The outputs approximate the complete recurrence above from
 * represented inputs. Internal state accumulation remains FP32; BF16 state storage widens on entry
 * and rounds once (RNE) on publication. Output storage is BF16. Scratch capacity is independent of
 * normalize_qk and state dtype.
 * `execution` supplies the stream and physical SM count for launch decomposition.
 * Inputs and out do not overlap state or one another. `ws` supplies transient storage reported by
 * gated_delta_net_workspace_capacity_bytes; scratch is scoped to the call. T may be any positive
 * value.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     Tensor& ssm_state, Tensor& out, CudaExecutionView execution);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state and must
 * match `ssm_state_in` in dtype (FP32 or BF16); `ssm_state_in` and `ssm_state_out` may be disjoint
 * or exactly the same storage. No other arguments may overlap either state.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     CudaExecutionView execution);

/**
 * One-token update for B independent records. q/k are contiguous BF16 [128,Hqk,1,B],
 * v/out BF16 [128,Hv,1,B], and g/beta FP32 [Hv,1,B]. Source and destination fields are dense
 * same-dtype FP32 or BF16 [128,128,Hv] within caller-owned records, selected by device I64 [B]
 * arena-relative byte offsets. B is positive. valid_columns is empty for dense rows or Device I32
 * [B] with values 0 or 1. A zero row reads no input values, offsets or state, writes exact BF16
 * zero output, and preserves every state field. Each live selected field must be 16-byte aligned
 * and fully inside its arena.
 * A row may update in place. Destinations are distinct and do not overwrite other active sources.
 * Input/output tensors do not overlap selected state fields. This form uses no arena allocation.
 */
void gated_delta_net_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  bool normalize_qk, ConstStateFieldView source_state,
                                  StateFieldView destination_state, const Tensor& valid_columns,
                                  Tensor& out, CudaExecutionView execution);

/**
 * Evaluates B independent normalized Gated DeltaNet recurrences from read-only initial_state.
 * q/k are BF16 [128,Hq,T,B], v/out BF16 [128,Hv,T,B], and g/beta FP32 [Hv,T,B].
 * initial_state selects dense FP32 or BF16 [128,128,Hv] fields using device I64 [B] record byte
 * offsets. Every live selected field is 16-byte aligned and in bounds. B>0, T=2..16; (Hq,Hv) is
 * one of (16,32), (16,48), (8,24). scale is 1/sqrt(128).
 *
 * valid_columns is empty for dense rows or device I32 [B] with values n[b] in [0,T]. Each row
 * evaluates n[b] steps. Its BF16 readouts approximate the mathematical recurrence above, and
 * private transition records bind that prefix to the initial state S0 for GdnReplayFoldPlan.
 * A zero row reads no q/k/v/g/beta, state offsets or state. Invalid record suffixes are
 * untouched and invalid output suffixes are exact BF16 +0. Inputs, selected state fields,
 * records, output and metadata are disjoint.
 */
void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   ConstStateFieldView initial_state, const Tensor& valid_columns,
                                   GdnRecurrenceRecordView records, Tensor& out,
                                   CudaExecutionView execution);


// Packed recurrence. Q/K are BF16 [128,Hqk,N], V/out BF16 [128,Hv,N], and G/beta
// FP32 [Hv,N]. Device I64 [S] byte offsets select independent same-dtype FP32 or BF16 [128,128,Hv]
// fields. The caller guarantees alignment to 16 bytes and disjoint destination fields.
// A destination may exactly alias its own source. Other selected state and write
// relationships follow gated_delta_net_batch_update.
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk,
                     const RaggedSequenceView& ragged, MemoryArena& ws,
                     ConstStateFieldView source_state, StateFieldView destination_state,
                     Tensor& out, CudaExecutionView execution);
[[nodiscard]] std::size_t gated_delta_net_ragged_workspace_capacity_bytes(
    std::int32_t qk_heads, std::int32_t value_heads, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t min_segments, std::int32_t max_segments);

} // namespace ninfer::ops
