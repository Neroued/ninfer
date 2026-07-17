#pragma once

// ninfer::ops - fused gate/up projection followed by SwiGLU.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the maximum transient capacity required by LinearSwiGLU for any T in [1,max_tokens].
 * `max_tokens` sizes caller-owned storage; it does not cap later Op calls when sufficient storage
 * is provided.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_bytes(std::int32_t gate_up_rows,
                                                        std::int32_t max_tokens);

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = Linear(x, gate_up_weight); M=gate_up_rows/2;
 *   out[i,t] = BF16(SiLU(gate_up[i,t]) * gate_up[M+i,t]).
 *
 * Logical shapes:
 *   x [5120,T] and out [17408,T] are contiguous BF16. gate_up_weight is Q4G64_F16S RowSplit
 *   [34816,5120] with FP16 scales. T may be any positive value.
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates the complete expression naively
 *   in FP64 before converting the observable result to BF16. Production routes may fuse or
 *   materialize gate/up and may choose their natural accumulator, staging, and workspace precision;
 *   those private choices are qualified directly against the same oracle.
 *
 * Effects:
 *   Writes the full output; x/weight and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_swiglu_workspace_bytes(34816,T), scoped to
 *   the call. There is no persistent state side effect.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

} // namespace ninfer::ops
