#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Computes four independent linear projections for each token:
 *
 *   q[:,t]    = linear(x[:,t], query_key_weight[0:6144,:])
 *   k[:,t]    = linear(x[:,t], query_key_weight[6144:7168,:])
 *   gate[:,t] = linear(x[:,t], gate_value_weight[0:6144,:])
 *   v[:,t]    = linear(x[:,t], gate_value_weight[6144:7168,:]).
 *
 * All tensors are contiguous BF16. Shapes are x [5120,T], q/gate [6144,T], and k/v [1024,T].
 * T may be any positive value.
 * The two parent weights are RowSplit [7168,5120] with FP16 scales and group size 64:
 * query_key is Q4G64_F16S and gate_value is Q5G64_F16S. The oracle exact-decodes each row and
 * evaluates every projection naively in FP64 before converting the observable outputs to BF16.
 * Production routes choose their private accumulator and staging precision. Inputs and the four
 * outputs must be mutually non-overlapping. Current registered routes require no transient
 * allocation; `ws` remains the Op-owned workspace boundary. The Op has no persistent state side
 * effect.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_weight,
                     const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                     WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
