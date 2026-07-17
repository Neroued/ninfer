#pragma once

// ninfer::ops - fused GDN Q/K/V input projections into one contiguous output.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Validates the registered problem and positive token capacity. Every admitted route writes
 * directly to the final output, so the required transient capacity is zero. `max_tokens` is a
 * capacity query, not an execution limit.
 */
[[nodiscard]] std::size_t gdn_input_proj_workspace_bytes(std::int32_t qk_rows,
                                                         std::int32_t value_rows,
                                                         std::int32_t max_tokens);

/**
 * Op: gdn_input_proj
 *
 * Math / indexing:
 *   qkv[:,t] = concat(qk_weight * x[:,t], v_weight * x[:,t]) for every token t.
 *
 * Logical shapes:
 *   x [5120,T], qk weight/output rows 4096, v weight/output rows 6144, qkv [10240,T].
 *   T may be any positive value.
 *   x and qkv are contiguous BF16. qk_weight is Q4G64_F16S RowSplit [4096,5120] and
 *   v_weight is Q5G64_F16S RowSplit [6144,5120], both with FP16 scales.
 *
 * Numeric:
 *   The oracle exact-decodes both weights and evaluates both projections naively in FP64 before
 *   converting the observable concatenated output to BF16. Production routes may choose their
 *   private precision independently; every registered route writes the final qkv allocation
 *   directly.
 *
 * Effects:
 *   Writes the full qkv output; inputs and output must not alias.
 *
 * Workspace:
 *   No transient bytes are required. The retained arena boundary is caller-owned and is not used.
 */
void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                    WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
