#pragma once

#include "core/device.h"
#include "ninfer/ops/attention_geometry.h"
#include "ninfer/ops/context_bucket.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Equal profiles permit capture reuse at fixed operands and workspace. Device positions,
// valid columns and record offsets remain replay inputs.
[[nodiscard]] OpGraphProfile sliding_window_attention_graph_profile(AttentionHeadGeometry geometry,
                                                                    std::uint32_t window,
                                                                    ContextBucket bucket,
                                                                    std::int32_t batch_size,
                                                                    std::int32_t tokens);

/**
 * Symmetric non-causal sliding-window grouped-query attention.
 *
 * Registered profiles are D=128, (Hq,Hkv)=(32,8) with window/cyclic capacity 2048 or 4096,
 * and (16,4) with capacity 2048; both use group 4, scale=1/sqrt(128), T=1..16, B>0.
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
 * Buckets K2 through M1 are supported. For every live row (V>0), the caller guarantees
 * 0 <= L <= context_bucket_tokens(bucket), L + V <= context_bucket_tokens(bucket), sequential live
 * positions, and that the selected record contains the declared committed interval. The bucket
 * determines launch and workspace capacity. The visible keys and numerical semantics follow the
 * device inputs. Fixed geometry and T/B preserve capture topology across bucket updates.
 */
void sliding_window_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                              const Tensor& positions, const Tensor& valid_columns,
                              AttentionHeadGeometry geometry, std::uint32_t window, float scale,
                              const ConstCyclicKVCacheLayerView& context, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

/**
 * Return transient capacity for every bucket and T in the inclusive ranges at the exact batch
 * size. Invalid profiles or ranges throw.
 */
[[nodiscard]] std::size_t sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window, ContextBucketRange buckets,
    std::int32_t batch_size, std::int32_t min_tokens, std::int32_t max_tokens);

} // namespace ninfer::ops
