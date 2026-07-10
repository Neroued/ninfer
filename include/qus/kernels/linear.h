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

} // namespace qus::kernels
