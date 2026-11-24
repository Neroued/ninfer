#pragma once

#include "ninfer/ops/ragged.h"

#include "ninfer/ops/attention_geometry.h"
#include "ninfer/ops/context_bucket.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::uint32_t kCausalAttentionMaximumVisibleKeys =
    context_bucket_tokens(ContextBucket::K256);

/**
 * Scaled dot-product attention over each entry's visible key set J:
 *   kvh = floor(h / (Hq/Hkv)); score[j] = scale * dot(q[:,h,i], K[:,kvh,j]);
 *   out[:,h,i] = BF16(sum_j softmax(score)[j] * V[:,kvh,j]).
 *
 * Inputs are interpreted in their represented dtype. BFloat16 caches store BF16 K and
 * FP16_RNE(BF16 input) V. Quantized rows follow kv_cache_append's persistent representation.
 * INT8/FP8 keys represent R*K, R=H256/16; their logits use R*q and decoded keys. NVFP4/K8V4
 * additionally store R*V and apply R^T after the complete value reduction. Newly appended rows
 * cross the persistent codec boundary before attention consumes them. Internal MMA operands,
 * softmax probabilities and partial sums are not additional public tensor boundaries.
 */

/**
 * Dense, non-causal single-segment attention.
 *
 * The registered profiles are D in {64,72}, Hq=Hkv=16, scale=1/sqrt(D). q/k/v are BF16 [D,16,T]
 * with contiguous feature and head dimensions; their token stride may be padded. out is contiguous
 * BF16 [D,16,T], with T>0. Every query attends all T keys. q/k/v/out are mutually non-overlapping,
 * inputs are unchanged, out is completely overwritten, and the Op has no persistent state side
 * effect. The single segment needs no transient workspace.
 */
void softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                       AttentionHeadGeometry geometry, float scale, MemoryArena& workspace,
                       Tensor& out, CudaExecutionView execution);

/**
 * Packed block-diagonal dense attention for the same D64/D72, H16 profiles.
 *
 * cu_seqlens is contiguous device I32 [S+1], starts at 0, ends at T, and is strictly increasing.
 * Each range [cu_seqlens[s],cu_seqlens[s+1]) is an independent non-causal segment; no score crosses
 * a segment boundary. q/k/v/out have the dtype, layout, alias, numerical, mutation, and state
 * contract of softmax_attention. Opaque tile descriptors are allocated from workspace for the
 * duration of the call; a single segment consumes no capacity.
 */
void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale, const Tensor& cu_seqlens,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

/**
 * Equal-length form of packed_softmax_attention. T is divisible by segment_length and consecutive
 * ranges [s*segment_length,(s+1)*segment_length) are the independent segments. Segment descriptors
 * are derived directly, so this form needs no workspace or descriptor-setup launch.
 */
void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale,
                              std::int32_t segment_length, Tensor& out,
                              CudaExecutionView execution);

/**
 * Return caller-owned transient capacity for every legal (T,S) pair in the inclusive envelope.
 * A pair is legal when 1 <= S <= T. An envelope with no legal pair throws; a legal single-segment
 * envelope may return zero.
 */
[[nodiscard]] std::size_t packed_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::int32_t min_tokens, std::int32_t max_tokens,
    std::int32_t min_segments, std::int32_t max_segments);

/**
 * Append K/V for B independent rows and compute causal grouped-query attention.
 *
 * The registered profiles are [D,Hq,Hkv]=[256,24,4] (group 6) and [256,16,2] (group 8), with
 * scale=1/sqrt(256). q/out are contiguous BF16 [D,Hq,W,B], k/v are contiguous BF16
 * [D,Hkv,W,B], positions are contiguous device I32 [W,B], kv_table_rows is contiguous device I32
 * [B], and the cache is BF16, INT8-G64, FP8-E4M3FN-G128/UE8M0, NVFP4-G16, or K8V4. valid_columns is
 * either contiguous device I32 [B] or an empty Tensor meaning every row has W live columns. This
 * dense/masked topology is chosen by the caller and never inferred by copying device metadata to
 * the host. B=1 accepts every positive W in the current prompt/decode domain; B>1 accepts
 * W=1..16.
 *
 * Let Vb be W for dense input or valid_columns[b] otherwise. For live column j<Vb with absolute
 * position p=positions[j,b], query head h attends cache rows [0,p] through table row
 * kv_table_rows[b]. The current k/v row is appended before it is observed, so the formula above
 * uses J=[0,p]. Each masked row has a live prefix [0,Vb); its live positions are sequential and
 * address populated histories. A nonempty row repeats its last live position through the inert
 * tail; an empty row does not access positions or the page-table selector. Other tail values are
 * safe dummies. Tail columns do not mutate cache and produce exact BF16 zero.
 *
 * Cache planes share packed page records selected by an I64 byte-offset table. Row and head
 * strides describe fields inside one record. Attention consumes this cache directly.
 * Caller-owned transient storage is bounded by the capacity query below; implementations that
 * need no partial state return zero capacity.
 *
 * Buckets K2 through K256 bound launch and workspace capacity. The caller guarantees
 * 0 <= p < min(context_bucket_tokens(bucket), logical_pages * 64) for every live column and
 * materialized page-table entries covering [0,p]. The page table can be smaller than the bucket.
 * Dense W must fit in the bucket. With a validity tensor only the live prefixes need to fit.
 * With fixed geometry, storage, B, W<=16, and mask presence, bucket updates preserve the graph's
 * node topology. Live device positions and validity determine the KV partitions on each replay.
 * Inputs, output, addressed cache fields, the offset table, and live workspace suballocations
 * are pairwise non-overlapping. The Op overwrites addressed cache rows. The caller owns cache
 * allocation and commit. Pass the same positive physical SM count from `execution` to the
 * workspace and graph-profile queries.
 */
void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

/**
 * Read-only single-sequence causal attention over an already populated cache.
 *
 * q/out are contiguous BF16 [256,24|16,T], positions is contiguous sequential device I32 [T],
 * and cache geometry, visible rows, scale, numerical semantics, bucket, alias, and workspace
 * rules are the same as causal_softmax_attention. The Op accepts no new K/V and leaves every cache
 * byte unchanged; out is completely overwritten.
 */
void causal_softmax_attention_cached(const Tensor& q, const Tensor& positions,
                                     AttentionHeadGeometry geometry, float scale,
                                     const ConstPagedKVLayerView& cache, ContextBucket bucket,
                                     MemoryArena& workspace, Tensor& out,
                                     CudaExecutionView execution);

/**
 * Return transient capacity for legal calls at the exact batch size across both inclusive
 * ranges of buckets and widths. Invalid profiles or ranges throw. The capacity may be zero.
 */
[[nodiscard]] std::size_t causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage, ContextBucketRange buckets,
    std::int32_t batch_size, std::int32_t min_tokens, std::int32_t max_tokens,
    std::int32_t multiprocessor_count);

// Equal profiles permit capture reuse at fixed tensor views, scale, and workspace base.
// Positions, valid columns, table rows, and page offsets are replay inputs.
[[nodiscard]] OpGraphProfile
causal_softmax_attention_graph_profile(AttentionHeadGeometry geometry, KvCacheStorage cache_storage,
                                       ContextBucket bucket, std::int32_t batch_size,
                                       std::int32_t tokens, std::int32_t multiprocessor_count,
                                       bool append = true, bool masked = false);

// Equal profiles permit capture reuse at fixed operands and workspace. Device metadata and
// page offsets remain replay inputs.
[[nodiscard]] OpGraphProfile context_softmax_attention_graph_profile(AttentionHeadGeometry geometry,
                                                                     ContextBucket bucket,
                                                                     std::int32_t batch_size,
                                                                     std::int32_t tokens);

/**
 * Non-causal grouped-query attention over persistent context plus one live query block.
 *
 * The registered profile is D=128, Hq=32, Hkv=8 (group 4), scale=1/sqrt(128), T=1..16, and
 * B>0. q/out are contiguous BF16 [128,32,T,B], query_k/query_v are contiguous BF16
 * [128,8,T,B], and context_lengths, valid_columns, and table_rows are contiguous device I32 [B].
 * The read-only BFloat16 context stores BF16 K and FP16 V fields [128,64,8] within each
 * packed page record. Its I64 [logical_pages,table_rows] table contains arena-relative byte
 * offsets; table_rows selects a row of that table.
 *
 * For row b, let L=context_lengths[b] and V=valid_columns[b]. Every live query i<V attends the
 * complete logical set consisting of context rows [0,L) followed by every temporary query K/V row
 * [0,V). There is no causal triangle. Columns i>=V are inert and produce exact BF16 zero. The
 * V=0 skips context lengths, page-table selectors, and all input/cache data. The
 * context and temporary query K/V remain separate physical segments; every input and cache byte is
 * unchanged, and out is the only observable mutation and is completely overwritten.
 *
 * Buckets K2 through K256 are supported. For every live row (V>0), the caller guarantees
 * 0 <= L <= min(context_bucket_tokens(bucket), logical_pages * 64) and materialized block table
 * entries for every logical page intersecting [0,L). The bucket determines launch and workspace
 * capacity. The visible keys and numerical semantics follow the device inputs. Fixed geometry
 * and T/B preserve capture topology across bucket updates. Inputs, context, output, and live
 * workspace allocations are pairwise non-overlapping.
 */
void context_softmax_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                               const Tensor& context_lengths, const Tensor& valid_columns,
                               const Tensor& table_rows, AttentionHeadGeometry geometry,
                               float scale, const ConstPagedKVBatchLayerView& context,
                               ContextBucket bucket, MemoryArena& workspace, Tensor& out,
                               CudaExecutionView execution);

/**
 * Return transient capacity for every bucket and T in the inclusive ranges at the exact batch
 * size. Invalid profiles or ranges throw. The returned capacity may be zero.
 */
[[nodiscard]] std::size_t context_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, ContextBucketRange buckets, std::int32_t batch_size,
    std::int32_t min_tokens, std::int32_t max_tokens);


/**
 * Packed causal append and attention. Q/out are BF16 [256,Hq,N], K/V are BF16 [256,Hkv,N].
 * positions is Device I32 [N], consecutive within each segment. Table rows are Device I32 [S].
 * The batch cache, per-segment context bucket and disjoint writes follow the append contract.
 */
void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const RaggedSequenceView& ragged, const Tensor& positions,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache, ContextBucket bucket,
                              MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

[[nodiscard]] std::size_t causal_softmax_attention_ragged_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage, ContextBucketRange buckets,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t min_segments,
    std::int32_t max_segments, std::int32_t multiprocessor_count);

} // namespace ninfer::ops
