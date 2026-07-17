#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the maximum transient capacity required by LinearAdd for any T in [1,max_tokens].
 * Dimensions must be positive. `max_tokens` sizes caller-owned storage; it does not cap later Op
 * calls when sufficient storage is provided.
 */
[[nodiscard]] std::size_t linear_add_workspace_bytes(std::int32_t output_rows,
                                                     std::int32_t input_rows,
                                                     std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   residual[:,t]' = BF16(residual[:,t] + Linear(x,w)[:,t]).
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Weight is either Q5G64_F16S RowSplit with
 *   logical shape [5120,17408] or [5120,6144], or W8G32_F16S RowSplit [2048,4096].
 *   T may be any positive value.
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates the complete expression naively
 *   in FP64 before converting the observable result to BF16. Production routes may fuse or
 *   materialize the projection and may choose their natural accumulator, staging, and workspace
 *   precision; those private choices are not semantic rounding boundaries and are qualified against
 *   the same oracle with the route-appropriate tolerance.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_bytes(N,K,T), scoped to the
 *   call. There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

} // namespace ninfer::ops
