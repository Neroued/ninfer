#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Computes four independent linear projections for each token:
 *
 *   q[:,t]    = linear(x[:,t], q_weight)
 *   gate[:,t] = linear(x[:,t], gate_weight)
 *   k[:,t]    = linear(x[:,t], k_weight)
 *   v[:,t]    = linear(x[:,t], v_weight).
 *
 * All tensors are contiguous BF16. Shapes are x [5120,T], q/gate [6144,T], and k/v [1024,T].
 * Weights are RowSplit with FP16 scales and group size 64: q and k are Q4G64_F16S, gate and v
 * are Q5G64_F16S, with logical [output_rows,5120] shapes. The oracle exact-decodes each weight and
 * evaluates every projection naively in FP64 before converting the observable outputs to BF16.
 * Production routes choose their private accumulator and staging precision. Inputs and the four
 * outputs must be mutually non-overlapping. Current registered routes require no transient
 * allocation; `ws` remains the Op-owned workspace boundary. The Op has no persistent state side
 * effect.
 */
void attn_input_proj(const Tensor& x, const Weight& q_weight, const Weight& gate_weight,
                     const Weight& k_weight, const Weight& v_weight, Tensor& q, Tensor& gate,
                     Tensor& k, Tensor& v, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
