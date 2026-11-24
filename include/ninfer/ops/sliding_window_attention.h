#pragma once

#include "core/device.h"
#include "ninfer/ops/attention_geometry.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"


#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SlidingWindowAttentionExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

/**
 * Symmetric non-causal sliding-window grouped-query attention.
 *
 * The registered profiles are D=128, Hq=32, Hkv=8 (group 4), window/cyclic capacity 2048 or
 * 4096, scale=1/sqrt(128), T=1..16, and B=1..8. q/out are contiguous BF16 [128,32,T,B],
 * query_k/query_v are contiguous BF16 [128,8,T,B], positions is contiguous device I32 [T,B], and
 * Context K/V select dense [128,padded_capacity,8] fields with contiguous Device I64 [B]
 * byte offsets. Every selected field is 16-byte aligned. valid_columns is Device I32 [B].
 * Row b has V=valid_columns[b] live query columns and positions[i,b]=L+i for i<V.
 * Context K/V offsets select its cyclic-cache records; inactive rows need no valid record. Columns
 * i>=V are an inert physical tail and produce exact BF16 zero.
 *
 * The read-only cyclic BFloat16 context stores BF16 K and FP16 V for committed absolute positions
 * [max(0,L-window),L), with absolute position p stored at slot p mod window. Temporary query K/V is
 * a separate BF16 segment at [L,L+V). For live query position p_i, a populated context or
 * temporary key at p_j is visible
 * exactly when abs(p_j-p_i)<window. Thus distance window-1 is included and distance window is
 * excluded; every live query also sees every live temporary query row in its batch row. Query
 * head h reads KV head floor(h/4).
 *
 * Over the visible set J, score[j]=scale*dot(q,K[j]) and out=BF16(sum_j softmax(score)[j]*V[j]).
 * Inputs are interpreted in their represented dtype, including BF16 temporary V and FP16 cached V.
 * q/query_k/query_v/positions/valid_columns, addressed context fields, out, and live workspace
 * allocations are pairwise non-overlapping. The K/V offset tables may share the same Tensor and
 * overlap no writable storage. Inputs and context remain unchanged; out is the only observable
 * mutation and is completely overwritten.
 *
 * The caller guarantees envelope.min_context <= L <= envelope.max_context, sequential nonnegative
 * live positions, and that the selected record contains the declared committed interval. The
 * envelope is a host launch/workspace resource promise; it never changes the admitted key set or
 * numerical result.
 */
void sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& positions, const Tensor& valid_columns,
                              AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                              const ConstCyclicKVCacheLayerView& context,
                              SlidingWindowAttentionExecutionEnvelope envelope,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

/**
 * Return transient capacity for every T in the inclusive optimized interval at the exact batch
 * size. Geometry, window, and execution envelope are fixed profile inputs; invalid inputs throw.
 */
[[nodiscard]] std::size_t sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window,
    SlidingWindowAttentionExecutionEnvelope envelope, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t batch_size);

} // namespace ninfer::ops
