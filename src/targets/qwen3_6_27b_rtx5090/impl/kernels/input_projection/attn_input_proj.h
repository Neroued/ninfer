#pragma once

// ninfer::kernels - fused full-attention Q/Gate/K/V input projections.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::kernels {

void attn_input_proj(const Tensor& x, const Weight& q_weight, const Weight& gate_weight,
                     const Weight& k_weight, const Weight& v_weight, Tensor& q, Tensor& gate,
                     Tensor& k, Tensor& v, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::kernels
