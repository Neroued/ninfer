#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/kda_replay_records.h"
#include "core/tensor.h"
#include "core/record_view.h"
#include "ninfer/ops/ragged.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Caller-owned transient capacity for every T in [min_tokens,max_tokens]. D=128,
 * Hqk>=1, Hv>=Hqk and Hv%Hqk==0. Returns zero when all calls use scratch-free recurrence.
 */
[[nodiscard]] std::size_t kimi_delta_attention_workspace_capacity_bytes(std::int32_t qk_heads,
                                                                        std::int32_t value_heads,
                                                                        std::int32_t min_tokens,
                                                                        std::int32_t max_tokens);

[[nodiscard]] std::size_t kimi_delta_attention_ragged_workspace_capacity_bytes(
    std::int32_t qk_heads, std::int32_t value_heads, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t min_segments, std::int32_t max_segments);

/**
 * Kimi Delta Attention, independently for every value head h. Its shared query/key head is
 * qh=floor(h/(Hv/Hqk)). Normalize each represented q/k row with epsilon 1e-6, then define
 *
 *   a_h          = exp(A_log[h])
 *   log_alpha[c] = lower_bound * sigmoid(a_h * (g[c,h,t] + dt_bias[c,h]))
 *   alpha[c]     = exp(log_alpha[c])
 *   b            = sigmoid(beta[h,t]).
 *
 * Starting from logical FP32 state S_h[value,key], for t in increasing order:
 *
 *   decayed      = S_h * diag(alpha)
 *   delta        = b * (v[:,h,t] - decayed * k[:,qh,t])
 *   S_h          = decayed + outer(delta, k[:,qh,t])
 *   ideal[:,h,t] = scale * S_h * q[:,qh,t].
 *
 * Contiguous represented inputs are q/k BF16 [128,Hqk,T], v/g/out BF16 [128,Hv,T],
 * beta BF16 [Hv,T], A_log FP32 [Hv], dt_bias FP32 [128,Hv], state FP32 or BF16 [128,128,Hv].
 * Physical state index is ((h*128+value)*128+key). Hqk>=1, Hv>=Hqk, Hv%Hqk==0, T>=1.
 * lower_bound is finite in [-5,0]; scale is 1/sqrt(128). Inputs/out are non-overlapping.
 * State input/output must share the same supported dtype (FP32 or BF16) and may be disjoint
 * or exactly alias; no other operand overlaps state.
 *
 * Internal running state stays FP32; published state uses the state tensor's storage dtype
 * (FP32 or BF16, rounded to nearest-even on write-back when BF16); output storage is BF16.
 * Results approximate the complete recurrence over the represented inputs.
 *
 * workspace supplies the capacity query above and is scoped to the call; the Op allocates no
 * device storage. execution supplies the stream and physical SM count for launch decomposition.
 * This overload publishes state after all T tokens into the same state storage.
 */
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace, Tensor& state,
                          Tensor& out, CudaExecutionView execution);

/** Distinct-state form of the same mathematical contract; exact state alias is also allowed. */
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, MemoryArena& workspace,
                          const Tensor& state_in, Tensor& state_out, Tensor& out,
                          CudaExecutionView execution);

/**
 * Packed ragged form over S = ragged.num_seqs() non-empty segments and N = ragged.total_tokens()
 * columns. q/k are BF16 [128,Hqk,N], v/g/out are BF16 [128,Hv,N], beta is BF16 [Hv,N], and
 * source_state/destination_state select S independent FP32 or BF16 (matching) [128,128,Hv] state
 * records by Device I64 [S] byte offsets. Alignment, bounds, and selected-state alias relationships
 * are caller guarantees, with the same rules as batch update. Same-segment exact in-place alias
 * is allowed, and workspace may be reused between segments on the execution stream.
 */
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, const RaggedSequenceView& ragged,
                          MemoryArena& workspace, ConstStateFieldView source_state,
                          StateFieldView destination_state, Tensor& out,
                          CudaExecutionView execution);

/**
 * Scratch-free one-token update for B independent state records, using the same head mapping.
 * q/k BF16 [128,Hqk,1,B], v/g/out BF16 [128,Hv,1,B], beta BF16 [Hv,1,B],
 * A_log FP32 [Hv], dt_bias FP32 [128,Hv], and source/destination select matching FP32 or BF16
 * [128,128,Hv] fields using device I64 [B] arena-relative byte offsets. Each field is 16-byte
 * aligned and in bounds;
 * other operands are contiguous. B is positive. valid_columns is empty for all active rows,
 * or Device I32 [B] with values in {0,1}. An inactive row writes exact BF16 +0 to output and
 * accesses neither its State offsets nor selected fields. Source/destination may exactly alias or
 * be disjoint. Destinations are distinct and cannot overwrite another row's source. Read-only
 * sources may be shared. Inputs, output and metadata do not overlap selected state fields.
 */
void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& A_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       ConstStateFieldView source, StateFieldView destination,
                                       const Tensor& valid_columns, Tensor& out,
                                       CudaExecutionView execution);

/**
 * Evaluate n[b] KDA steps from each read-only initial state S0, publishing BF16 readouts
 * that approximate the mathematical recurrence above. The opaque transition records bind
 * the valid prefix to that same S0 for KdaReplayFoldPlan.
 *
 * Inputs use the batch-update shapes with T>=1 and B>=1. valid_columns is empty (n=T) or
 * Device I32 [B], 0<=n<=T. Invalid output tails are exact BF16 +0. Invalid record suffixes
 * and initial state are unchanged. Zero-length rows access neither State selectors nor fields.
 * Record planes are disjoint from one another, all inputs, output, state and metadata.
 * Device selectors/lengths are caller promises. Zero scratch; capture/replay reads current
 * I64 source offsets and valid lengths at stable metadata addresses.
 */
void kimi_delta_attention_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& g, const Tensor& beta, const Tensor& A_log,
                                        const Tensor& dt_bias, float lower_bound, float scale,
                                        ConstStateFieldView initial_state,
                                        const Tensor& valid_columns, KdaRecurrenceRecordView records,
                                        Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
