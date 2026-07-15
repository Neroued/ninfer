#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns zero for T=1 or T>16; otherwise returns one contiguous BF16
 * [output_rows,tokens] fallback matrix. Dimensions must be positive.
 */
[[nodiscard]] std::size_t linear_add_workspace_bytes(std::int32_t output_rows,
                                                     std::int32_t input_rows, std::int32_t tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   residual[:,t]' = bf16(float(residual[:,t]) + float(linear(x,w)[:,t])).
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [5120,T]. Weight is Q5G64_F16S RowSplit with FP16
 *   scales and logical shape [5120,17408] or [5120,6144].
 *
 * Numeric:
 *   The linear projection follows linear.h. Fused implementations preserve the registered
 *   projection-plus-residual BF16 rounding seam.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_bytes(5120,K,T), scoped to
 *   the call. There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

} // namespace ninfer::ops
