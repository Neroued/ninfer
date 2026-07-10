#pragma once

// qus::kernels - linear: out[:, t] = W @ x[:, t].
// Precision seam op: dense BF16_CTRL/FP32_CTRL weights run through as_dense(w);
// quantized text qtypes are wired by later tasks.

#include "qus/core/arena.h"
#include "qus/core/tensor.h"

#include <cuda_runtime.h>

namespace qus::kernels {

// x: [K,T] BF16, w: [N,K] Weight, out: [N,T] BF16. Fastest dim first.
void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws, cudaStream_t stream);

// MTP prefill-only paired W8G32 K/V projection. Small-T falls back to two
// ordinary linears; Large-T shares the activation staging in one Tensor Core
// kernel.
void linear_w8g32_kv_pair(const Tensor& x, const Weight& k_weight, const Weight& v_weight,
                          Tensor& k_out, Tensor& v_out, WorkspaceArena& ws, cudaStream_t stream);

// Text full-attention prefill input projection.  Large-T maps the four Q4/Q5
// projections into one CTA-grouped launch; Small-T retains the normal linear
// dispatch.
void linear_attn_input_grouped(const Tensor& x, const Weight& q_weight, const Weight& gate_weight,
                               const Weight& k_weight, const Weight& v_weight, Tensor& q,
                               Tensor& gate, Tensor& k, Tensor& v, WorkspaceArena& ws,
                               cudaStream_t stream);

// Text GDN prefill q/k/v projection. qk_weight is the existing fused Q4
// [4096,5120] block; the grouped kernel writes Q4 q/k and Q5 v directly into
// the contiguous [10240,T] qkv tensor.
void linear_gdn_input_grouped(const Tensor& x, const Weight& qk_weight, const Weight& v_weight,
                              Tensor& qkv, WorkspaceArena& ws, cudaStream_t stream);

} // namespace qus::kernels
