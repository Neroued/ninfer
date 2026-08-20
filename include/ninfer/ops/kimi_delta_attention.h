#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Applies the recurrent Kimi Delta Attention transition independently for every head. Query, key,
 * value, and the raw per-key gate all have the same head count H. For each head h and token t,
 * let q and k be L2-normalized with epsilon 1e-6, and define
 *
 *   a_h          = exp(A_log[h])
 *   log_alpha[c] = lower_bound * sigmoid(a_h * (g[c,h,t] + dt_bias[c,h]))
 *   alpha[c]     = exp(log_alpha[c])
 *   b            = sigmoid(beta[h,t]).
 *
 * With logical FP32 state S_h[value,key], the transition and updated-state readout are
 *
 *   decayed       = S_h * diag(alpha)
 *   delta         = b * (v[:,h,t] - decayed * k)
 *   S_h           = decayed + outer(delta, k)
 *   ideal[:,h,t]  = scale * S_h * q.
 *
 * Shapes/dtypes are contiguous q/k/v/g/out BF16 [128,H,T], beta BF16 [H,T], A_log FP32 [H],
 * dt_bias FP32 [128,H], and state FP32 [128,128,H]. The physical state index is
 * ((h * 128 + value) * 128 + key). H and T may be any positive values. `lower_bound` is finite and
 * in [-5,0], and `scale` is 1/sqrt(128). Inputs and out do not overlap state or one another.
 *
 * The mathematical oracle evaluates the complete formula naively in FP64 from the represented
 * BF16 inputs and FP32 parameters/state. Intermediate arithmetic and transcendental
 * approximations are private implementation details; the published state remains FP32 and out is
 * rounded once to BF16.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, Tensor& ssm_state, Tensor& out,
                          cudaStream_t stream);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` may be disjoint or exactly the same storage, but may not
 * partially overlap. No other argument may overlap either state.
 */
void kimi_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                          const Tensor& beta, const Tensor& A_log, const Tensor& dt_bias,
                          float lower_bound, float scale, const Tensor& ssm_state_in,
                          Tensor& ssm_state_out, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
