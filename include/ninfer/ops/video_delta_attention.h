#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Semantic local window, independent of execution tiling. Input frame f belongs to
// floor((f+frame_offset)/chunk_frames); exclude that chunk and radius neighbours.
struct VideoDeltaAttentionWindow {
    std::int32_t chunk_frames = 5;
    std::int32_t radius       = 1;
    std::int32_t frame_offset = 0;
};

/**
 * Transient arena capacity covering every F in [min_frames,max_frames], including both
 * readout variants. H=1..56, U>=1, B>=1, F>=0 and F*U fits int32. C>=1, radius>=0,
 * offset in [0,C). No device allocation or data-dependent query. Includes alignment slack.
 */
[[nodiscard]] std::size_t video_delta_attention_workspace_capacity_bytes(
    std::int32_t heads, std::int32_t tokens_per_frame, std::int32_t min_frames,
    std::int32_t max_frames, std::int32_t batch_size, VideoDeltaAttentionWindow window);

[[nodiscard]] std::size_t video_delta_state_init_workspace_capacity_bytes(std::int32_t heads,
                                                                          std::int32_t min_tokens,
                                                                          std::int32_t max_tokens,
                                                                          std::int32_t batch_size);

/**
 * Zero-prior weighted KV fit, independently per head and batch:
 * A=sum_i beta_i k_i k_i^T, B=sum_i beta_i v_i k_i^T;
 * state=state_scale * B (I+A)^-1. state_scale is finite; L=0 writes exact FP32 +0.
 * k/v are prepared BF16 [128,H,L,B], beta is FP32 [H,L,B]. Output state is contiguous
 * FP32 [key=128,value=128,H,B], physical index ((b*H+h)*128+value)*128+key.
 * The vector stride, data validity, ownership and asynchronous rules below apply.
 */
void video_delta_state_init(const Tensor& k, const Tensor& v, const Tensor& beta, float state_scale,
                            MemoryArena& workspace, Tensor& state, CudaExecutionView execution);

/**
 * Bidirectional frame-wise implicit delta attention. D=128, H=1..56, B>=1, F>=0,
 * U=tokens_per_frame>=1, T=F*U. q/k/v/out BF16 [128,H,T,B], alpha FP32 [128,H,F,B],
 * beta FP32 [H,T,B], initial_state FP32 [key=128,value=128,H,B]. Hq=Hk=Hv.
 * q/k/v are already activated/normalized features; no RoPE, L2Norm or 1/sqrt(D) is added.
 * All represented inputs are finite, alpha/beta in [0,1], mathematical intermediates
 * representable. Values are caller promises, not device-to-host validation.
 *
 * Independently per (b,h), A_f=sum_u beta_fu k_fu k_fu^T, B_f=sum_u beta_fu v_fu k_fu^T.
 * S_f^+=(S_(f-1)^+ diag(alpha_f)+B_f)(I+A_f)^-1, S_-1^+=initial_state.
 * S_f^-=(S_(f+1)^- diag(alpha_f)+B_f)(I+A_f)^-1, S_F^-=initial_state.
 * For c=floor((f+offset)/C), lo=max(0,(c-radius)*C-offset),
 * hi=min(F-1,(c+radius+1)*C-offset-1), left=S_(lo-1)^+, right=S_(hi+1)^-.
 * Out-of-range sides use initial_state. With a=max(alpha,1e-12),
 * ideal_fu=(left diag(prod_(lo..f) a)+right diag(prod_(f..hi) a)) q_fu.
 * Clamp applies only to bridge, not recurrence. Initial state is already scaled (H3: half
 * the text fit), remains unchanged, and is not a persistent decoding cache.
 *
 * q/k/v (and gate_logits below) permit positive, non-overlapping, ordered outer strides;
 * nb[0]=sizeof(BF16), each following axis starts no earlier than the preceding span.
 * Natural dtype alignment suffices. Other operands are contiguous. Read-only operands may
 * share backing; output and live scratch are disjoint from every input and one another.
 * F=0 submits no device work. Nonempty output is completely overwritten. Results approximate
 * the complete formula from represented inputs; private reduction/casting order is not an ABI.
 * Factorization and master state arithmetic use FP32; private contraction packing is not an ABI.
 * All backing belongs to execution.device().
 *
 * workspace is borrowed only in a scope; no hidden allocation, synchronization or retained
 * scratch. Caller keeps all backing alive until stream completion and orders cross-stream reuse.
 * Fixed geometry/strides/addresses capture; replay reads current operand values. Interval
 * capacity does not make a captured graph variable-shape. No collective or model policy.
 */
void video_delta_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& alpha,
                           const Tensor& beta, const Tensor& initial_state,
                           std::int32_t tokens_per_frame, VideoDeltaAttentionWindow window,
                           MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

/**
 * Same readout followed by y_d=o_d*rsqrt(mean_j(o_j^2)+eps)*weight_d*sigmoid(z_d).
 * norm_weight is BF16 [128], gate_logits BF16 [128,H,T,B], eps positive and finite.
 * Norm is per token/head. No required BF16 readout intermediate; verify the complete formula,
 * not bitwise composition of the two public operators. Other rules and capacity query above apply.
 */
void video_delta_attention_sigmoid_gated(const Tensor& q, const Tensor& k, const Tensor& v,
                                         const Tensor& alpha, const Tensor& beta,
                                         const Tensor& initial_state, const Tensor& norm_weight,
                                         const Tensor& gate_logits, float norm_eps,
                                         std::int32_t tokens_per_frame,
                                         VideoDeltaAttentionWindow window, MemoryArena& workspace,
                                         Tensor& out, CudaExecutionView execution);

} // namespace ninfer::ops
