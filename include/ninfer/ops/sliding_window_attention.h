#pragma once

#include "core/device.h"
#include "ninfer/ops/attention_geometry.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct SlidingWindowAttentionExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

// Equality means the captured call is reusable for another envelope at fixed operands.
// Device positions, valid columns and State offsets remain replay inputs.
class SlidingWindowAttentionGraphProfile {
public:
    bool operator==(const SlidingWindowAttentionGraphProfile&) const = default;
private:
    std::array<std::int32_t, 11> words_{};
    friend SlidingWindowAttentionGraphProfile
        sliding_window_attention_graph_profile(AttentionHeadGeometry, std::uint32_t,
                                               SlidingWindowAttentionExecutionEnvelope,
                                               std::int32_t, std::int32_t);
};

[[nodiscard]] SlidingWindowAttentionGraphProfile
sliding_window_attention_graph_profile(AttentionHeadGeometry, std::uint32_t window,
                                       SlidingWindowAttentionExecutionEnvelope, std::int32_t tokens,
                                       std::int32_t batch);

/**
 * Symmetric non-causal sliding-window grouped-query attention.
 *
 * Registered profiles are D=128, (Hq,Hkv)=(32,8) with window/cyclic capacity 2048 or 4096,
 * and (16,4) with capacity 2048; both use group 4, scale=1/sqrt(128), T=1..16, B=1..8.
 * q/out are contiguous BF16 [128,Hq,T,B], query_k/query_v are contiguous BF16 [128,Hkv,T,B],
 * positions is contiguous Device I32 [T,B], and Context K/V select dense
 * [128,padded_capacity,Hkv] fields with contiguous Device I64 [B]
 * byte offsets. Every selected field is 16-byte aligned. valid_columns is Device I32 [B].
 * Row b has V=valid_columns[b] live query columns and positions[i,b]=L+i for i<V.
 * Context K/V offsets select its cyclic-cache records; inactive rows need no valid record. Columns
 * i>=V are an inert physical tail and produce exact BF16 zero. A row with V=0 does not read
 * positions, query K/V, or record-offset values.
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
 * numerical result. Fixed geometry and T/B preserve capture topology across envelope updates.
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
