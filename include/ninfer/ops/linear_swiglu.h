#pragma once

// ninfer::ops - fused gate/up projection followed by SwiGLU.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns zero for T=1 or T>16; otherwise returns one contiguous BF16
 * [gate_up_rows,tokens] fallback matrix. gate_up_rows must be positive and even.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_bytes(std::int32_t gate_up_rows,
                                                        std::int32_t tokens);

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = linear(x, gate_up_weight); M=gate_up_rows/2;
 *   out[i,t] = bf16(silu(float(gate_up[i,t])) * float(gate_up[M+i,t])).
 *
 * Logical shapes:
 *   x [5120,T] and out [17408,T] are contiguous BF16. gate_up_weight is Q4G64_F16S RowSplit
 *   [34816,5120] with FP16 scales.
 *
 * Numeric:
 *   Fused paths preserve the registered projection/SwiGLU rounding boundary.
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
