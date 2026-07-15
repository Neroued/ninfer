#pragma once

// ninfer::ops - fused GDN Q/K/V input projections into one contiguous output.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns two contiguous BF16 temporary matrices, [qk_rows,tokens] and
 * [value_rows,tokens], for 1<=tokens<=16; returns zero for tokens>16.
 */
[[nodiscard]] std::size_t
gdn_input_proj_workspace_bytes(std::int32_t qk_rows, std::int32_t value_rows, std::int32_t tokens);

/**
 * Op: gdn_input_proj
 *
 * Math / indexing:
 *   qkv[:,t] = concat(qk_weight * x[:,t], v_weight * x[:,t]) for every token t.
 *
 * Logical shapes:
 *   x [5120,T], qk weight/output rows 4096, v weight/output rows 6144, qkv [10240,T].
 *   x and qkv are contiguous BF16. qk_weight is Q4G64_F16S RowSplit [4096,5120] and
 *   v_weight is Q5G64_F16S RowSplit [6144,5120], both with FP16 scales.
 *
 * Numeric:
 *   Each projection follows linear.h; the concatenated BF16 outputs preserve their projection
 *   rounding boundaries.
 *
 * Effects:
 *   Writes the full qkv output; inputs and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by gdn_input_proj_workspace_bytes(4096,6144,T),
 *   scoped to the call. There is no persistent state side effect.
 */
void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                    WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
