#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// 1) Precompute step-invariant f_rnn contribution ONCE per batch row + per-step hidden
// contribution:
//    base[c, b]           = FP32(bias[c]) + dot(w_frnn[c, :], f_rnn[:, b])
//    context[c, b, step]  = base[c, b] + dot(w_hidden[c, :], hidden[:, step, b])   (FP32)
//    hidden: BF16 [H, D_steps, B], f_rnn: BF16 [H, B], w_hidden/w_frnn: Weight BF16 Contiguous
//    [3*r, H], bias: FP32 [3*r], context: FP32 [3*r, B, D_steps] (step-outer so each step is a
//    contiguous [3*r, B] slice). D_steps >= 1, B >= 1, H % 8 == 0, r % 16 == 0.
[[nodiscard]] std::size_t markov_rnn_context_workspace_capacity_bytes(std::int32_t hidden,
                                                                      std::int32_t rank,
                                                                      std::int32_t max_steps,
                                                                      std::int32_t max_batch);

void markov_rnn_context(const Tensor& hidden, const Tensor& f_rnn, const Weight& w_hidden,
                        const Weight& w_frnn, const Tensor& bias, MemoryArena& workspace,
                        Tensor& context, CudaExecutionView execution);

inline constexpr std::int32_t kMarkovRnnStepMaxInPlaceRank = 65536;

// 2) Single fused recurrent step (embed gather + [3*r, 2*r] addmm + exact BF16-boundary
// markov_gate):
//    For each live row b in [0, B):
//      prev_emb = w1_embed[:, prev_tokens[b]]                                   (BF16 [r])
//      raw[0..3r-1] = context_step[:, b] + w_recurrent @ [state_in[:, b]; prev_emb]  (FP32)
//      gate = RN_BF16(raw[0..r-1]), cand = RN_BF16(raw[r..2r-1]), out_r = RN_BF16(raw[2r..3r-1])
//      g    = RN_BF16(1.0f / (1.0f + expf(-FP32(gate))))
//      c    = RN_BF16(tanhf(FP32(cand)))
//      state_out[:, b] = RN_BF16(FP32(RN_BF16(FP32(g) * FP32(state_in[:, b])))
//                              + FP32(RN_BF16(FP32(RN_BF16(1.0f - FP32(g))) * FP32(c))))
//      latent[:, b]    = RN_BF16(tanhf(FP32(out_r)))
//    Inactive rows (live_rows non-empty && live_rows[b] == 0): write exact BF16 zero to state_out
//    and latent without reading prev_tokens / w1_embed / state_in / context_step.
//    state_in/state_out/latent: BF16 [r, B]. Exact aliasing of state_out and state_in is supported
//    for r <= kMarkovRnnStepMaxInPlaceRank (65536). Larger ranks require distinct state buffers.
//    prev_tokens: I32 [B]; context_step: FP32 [3*r, B]; w1_embed: BF16 [r, V_pad];
//    w_recurrent: Weight BF16 Contiguous [3*r, 2*r]; live_rows: empty or I32 [B]. Zero workspace.
void markov_rnn_step(const Tensor& state_in, const Tensor& prev_tokens, const Tensor& context_step,
                     const Tensor& w1_embed, const Weight& w_recurrent, const Tensor& live_rows,
                     Tensor& state_out, Tensor& latent, CudaExecutionView execution);

// 3) Fused low-rank folded LM-head delta + base_logits + softcap:
//    For each live row b in [0, B) and v in [0, token_domain):
//      delta = RN_BF16(dot(folded_weight[v, :], latent[:, b]) + FP32(folded_bias[v]))
//      sum   = RN_BF16(FP32(base_logits[v, b]) + FP32(delta))
//      logits_out[v, b] = RN_BF16(softcap * tanhf(FP32(sum) / softcap))
//    For v in [token_domain, V_phys) or inactive rows (live_rows[b] == 0): exact BF16 zero.
//    latent: BF16 [r, B]; folded_weight: Weight BF16 Contiguous [V_phys, r]; folded_bias: BF16
//    [V_phys]; base_logits/logits_out: BF16 [V_phys, B] (logits_out may exactly alias base_logits);
//    1 <= token_domain <= V_phys; softcap > 0 finite; live_rows: empty or I32 [B]. Zero workspace.
void markov_logits_add_softcap(const Tensor& latent, const Weight& folded_weight,
                               const Tensor& folded_bias, const Tensor& base_logits,
                               std::int32_t token_domain, float softcap, const Tensor& live_rows,
                               Tensor& logits_out, CudaExecutionView execution);

} // namespace ninfer::ops
