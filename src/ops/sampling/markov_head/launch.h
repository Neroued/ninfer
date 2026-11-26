#pragma once

#include "ninfer/ops/markov_head.h"

namespace ninfer::ops::markov {

inline constexpr int kContextKTile = 512;

void launch_context(const Tensor& hidden, const Tensor& f_rnn, const Weight& w_hidden,
                    const Weight& w_frnn, const Tensor& bias, float* partial, Tensor& context,
                    CudaExecutionView execution);

void launch_step(const Tensor& state_in, const Tensor& prev_tokens, const Tensor& context_step,
                 const Tensor& w1_embed, const Weight& w_recurrent, const Tensor& live_rows,
                 Tensor& state_out, Tensor& latent, CudaExecutionView execution);

void launch_logits(const Tensor& latent, const Weight& folded_weight, const Tensor& folded_bias,
                   const Tensor& base_logits, int domain, float softcap, const Tensor& live_rows,
                   Tensor& logits_out, CudaExecutionView execution);

} // namespace ninfer::ops::markov
