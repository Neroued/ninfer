#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/record_view.h"

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
 * beta BF16 [Hv,T], A_log FP32 [Hv], dt_bias FP32 [128,Hv], state FP32 [128,128,Hv].
 * Physical state index is ((h*128+value)*128+key). Hqk>=1, Hv>=Hqk, Hv%Hqk==0, T>=1.
 * lower_bound is finite in [-5,0]; scale is 1/sqrt(128). Inputs/out are non-overlapping.
 * State input/output may be disjoint or exactly alias; no other operand overlaps state.
 *
 * Running and published state remain FP32; output storage is BF16. Results approximate the
 * complete recurrence over the represented inputs.
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
 * Scratch-free one-token update for B independent state records, using the same head mapping.
 * q/k BF16 [128,Hqk,1,B], v/g/out BF16 [128,Hv,1,B], beta BF16 [Hv,1,B],
 * A_log FP32 [Hv], dt_bias FP32 [128,Hv], and states selects dense FP32 [128,128,Hv] fields
 * using device I64 [B] arena-relative byte offsets. Each field is 16-byte aligned and in bounds;
 * other operands are contiguous. B is in [1,8]. Selected fields are distinct and updated in
 * place. Inputs, output and metadata do not overlap selected state fields.
 */
void kimi_delta_attention_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                       const Tensor& g, const Tensor& beta, const Tensor& A_log,
                                       const Tensor& dt_bias, float lower_bound, float scale,
                                       StateFieldView states, Tensor& out,
                                       CudaExecutionView execution);

} // namespace ninfer::ops
