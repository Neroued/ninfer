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

/**
 * Qwen3.6-35B W8 specialization. The one W8G32_F16S RowSplit parent has shape [9216,2048]
 * and stored row order [query 4096, key 512, gate 4096, value 512]. `x` is contiguous BF16
 * [2048,T], q/gate are contiguous BF16 [4096,T], and k/v are contiguous BF16 [512,T]. Every
 * route writes the four independent final allocations directly; no parent output or transient
 * workspace is materialized. T may be any positive value. The observable outputs use the same
 * independent exact-decode oracle and BF16 boundary described above.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, WorkspaceArena& ws, cudaStream_t stream);

/**
 * Qwen3.6 companion W8 specialization. The W8G32_F16S RowSplit parent has shape [6144,2048]
 * and stored row order [query 4096, key 1024, value 1024]. `x` is contiguous BF16 [2048,T],
 * q is contiguous BF16 [4096,T], and k/v are contiguous BF16 [1024,T]. Every route writes
 * the three independent final allocations directly; no parent output or transient workspace
 * is materialized. T may be any positive value. Q and K remain raw projection outputs: this
 * Op does not normalize or rotate either tensor.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_value_weight, Tensor& q, Tensor& k,
                     Tensor& v, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
