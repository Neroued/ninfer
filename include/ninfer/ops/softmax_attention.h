#pragma once

#include "ninfer/ops/attention_geometry.h"

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::uint32_t kCausalAttentionMaximumVisibleKeys = 262144;

struct CausalAttentionExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
};

struct ContextAttentionExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

/**
 * Shared numerical contract.
 *
 * Every entry computes stable scaled dot-product Softmax Attention. Query head h reads KV head
 * floor(h / (Hq/Hkv)). Public BF16 inputs and persistent cache rows are interpreted after their
 * storage boundary. In the BFloat16 cache profile, K is stored as BF16 and V as
 * FP16_RNE(BF16 input). For a declared visible key set J, the independent mathematical oracle is
 *
 *   score[j]       = scale * dot(FP64(q[:,h,i]), FP64(k[:,kvh,j]))
 *   probability[j] = exp(score[j] - max(score)) / sum_x exp(score[x] - max(score))
 *   ideal[:,h,i]   = sum_j probability[j] * FP64(v[:,kvh,j]).
 *
 * Dot products, stable Softmax, and the value reduction are evaluated naively in FP64. The BF16
 * Op output is promoted to FP64 for comparison; storage rounding belongs to the Op criterion and
 * is not reproduced by the oracle.
 *
 * Quantized cache rows use the exact represented values defined by kv_cache_append. Quantized K
 * is consumed through its matching Q/K profile. NVFP4 and K8V4 store R*V for the normalized
 * Hadamard R=H256/16 and apply R^T after the complete attention reduction. Their represented-value
 * conformance oracle independently evaluates RQ, RK, and RV in FP32. K8V4 applies the exact FP8
 * Q/K codecs; NVFP4 rounds RQ once to FP16 and exactly expands the persistent NVFP4 K/V
 * representation to FP16. The oracle then evaluates QK, stable Softmax, PV, and R^T in FP64 and
 * compares the final BF16 production output. Its delta from the unquantized BF16/FP64 oracle above
 * is separate quantization-quality evidence.
 *
 * The qualified BFloat16 compute profile keeps Q/K and persistent K at BF16 and uses native BF16
 * QK plus FP16 P/V MMA. INT8-G64 uses native signed-INT8 Q/K MMA; its prompt route uses FP16 P/V
 * MMA and its small-T route uses BF16 P/V MMA. FP8 and K8V4 use native E4M3FN QK MMA, while NVFP4
 * uses FP16 Q and exactly expanded FP16 K with native FP16 QK MMA in both prompt and small-T
 * routes. NVFP4 never quantizes Q to FP4 or FP8. INT8 QK accumulates each group in INT32 and
 * combines represented group products in FP32; the other QK profiles accumulate in FP32. Every
 * profile retains FP32 accumulation for PV, split state, merge, normalization, and applicable
 * Hadamard reductions. P is never quantized to FP8/FP4, and only the final public output is stored
 * as BF16. These arithmetic paths are implementation profiles rather than extra public tensor
 * boundaries. Every cache route has one named numerical criterion and is checked directly against
 * its independent oracle; route-to-route parity is only supplementary evidence.
 * Those criteria apply to the registered geometries, tested extents, conformance matrix, and
 * target-representative activation range; they are not universal error bounds for arbitrary
 * adversarial BF16 tensors.
 */

/**
 * Dense, non-causal single-segment attention.
 *
 * The registered profile is D=72, Hq=Hkv=16, scale=1/sqrt(72). q/k/v are BF16 [72,16,T]
 * with contiguous feature and head dimensions; their token stride may be padded. out is contiguous
 * BF16 [72,16,T]. Every query attends all T keys. q/k/v/out are mutually non-overlapping, inputs
 * are unchanged, out is completely overwritten, and the Op has no persistent state side effect.
 * The single segment needs no transient workspace.
 */
void softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                       AttentionHeadGeometry geometry, float scale, WorkspaceArena& workspace,
                       Tensor& out, cudaStream_t stream);

/**
 * Packed block-diagonal dense attention for the same D72/H16 profile.
 *
 * cu_seqlens is contiguous device I32 [S+1], starts at 0, ends at T, and is strictly increasing.
 * Each range [cu_seqlens[s],cu_seqlens[s+1]) is an independent non-causal segment; no score crosses
 * a segment boundary. q/k/v/out have the dtype, layout, alias, numerical, mutation, and state
 * contract of softmax_attention. Opaque tile descriptors are allocated from workspace for the
 * duration of the call; a single segment consumes no capacity.
 */
void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale, const Tensor& cu_seqlens,
                              WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Equal-length form of packed_softmax_attention. T is divisible by segment_length and consecutive
 * ranges [s*segment_length,(s+1)*segment_length) are the independent segments. Segment descriptors
 * are derived directly, so this form needs no workspace or descriptor-setup launch.
 */
void packed_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              AttentionHeadGeometry geometry, float scale,
                              std::int32_t segment_length, Tensor& out, cudaStream_t stream);

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
 * [B], and the cache is BF16, INT8-G64, row-scaled FP8-E4M3FN, NVFP4-G16, or K8V4. valid_columns is
 * either contiguous device I32 [B] or an empty Tensor meaning every row has W live columns. This
 * dense/masked topology is chosen by the caller and never inferred by copying device metadata to
 * the host. B=1 accepts every positive W in the current prompt/decode domain; B=2..8 accepts
 * W=1..16.
 *
 * Let Vb be W for dense input or valid_columns[b] otherwise. For live column j<Vb with absolute
 * position p=positions[j,b], query head h attends cache rows [0,p] through table row
 * kv_table_rows[b]. The current k/v row is appended before it is observed, so the formula is the
 * shared oracle above over J=[0,p]. Each masked row has a live prefix [0,Vb); its live positions
 * are sequential and address populated histories. A nonempty row repeats its last live position
 * through the inert tail; an empty row uses zero positions. Other tail values are safe dummies.
 * Tail columns do not mutate cache and produce exact BF16 zero.
 *
 * The registered prompt route consumes the paged cache directly and requires zero transient
 * workspace. Small-T routes may use the split state returned by the capacity query below.
 *
 * The caller guarantees that the maximum p+1 over live rows lies within envelope. The envelope is
 * a host launch/workspace resource promise over that batch maximum, not a mask and not persistent
 * state. Inputs, output, every cache plane/table, and live workspace suballocations are pairwise
 * non-overlapping. The Op overwrites every addressed cache row but owns no cache allocation,
 * frontier, request identity, or commit authority.
 */
void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, WorkspaceArena& workspace,
                              Tensor& out, cudaStream_t stream);

/**
 * Read-only single-sequence causal attention over an already populated cache.
 *
 * q/out are contiguous BF16 [256,24|16,T], positions is contiguous sequential device I32 [T],
 * and cache geometry, visible rows, scale, numerical oracle, envelope, alias, and workspace rules
 * are the same as causal_softmax_attention. The Op accepts no new K/V and leaves every cache byte
 * unchanged; out is completely overwritten.
 */
void causal_softmax_attention_cached(const Tensor& q, const Tensor& positions,
                                     AttentionHeadGeometry geometry, float scale,
                                     const PagedKVLayerView& cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Return transient capacity for every W in the inclusive interval at one exact batch size. The
 * head geometry, cache dtype, and execution envelope are fixed implementation-profile inputs.
 * Invalid profiles or intervals throw; an interval containing only prompt routes returns zero.
 */
[[nodiscard]] std::size_t causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage,
    CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size, std::int32_t min_tokens,
    std::int32_t max_tokens);

/**
 * Non-causal grouped-query attention over persistent context plus one live query block.
 *
 * The registered profile is D=128, Hq=32, Hkv=8 (group 4), scale=1/sqrt(128), T=1..16, and
 * B=1..8. q/out are contiguous BF16 [128,32,T,B], query_k/query_v are contiguous BF16
 * [128,8,T,B], and context_lengths, valid_columns, and table_rows are contiguous device I32 [B].
 * The read-only paged BFloat16 context uses head-major BF16 K and FP16 V planes
 * [128,64,Nphysical,8].
 *
 * For row b, let L=context_lengths[b] and V=valid_columns[b]. Every live query i<V attends the
 * complete logical set consisting of context rows [0,L) followed by every temporary query K/V row
 * [0,V). There is no causal triangle. Columns i>=V are inert and produce exact BF16 zero. The
 * context and temporary query K/V remain separate physical segments; every input and cache byte is
 * unchanged, and out is the only observable mutation and is completely overwritten.
 *
 * The caller guarantees envelope.min_context <= L <= envelope.max_context and materialized block
 * table entries for every logical page intersecting [0,L). The envelope may affect finite launch
 * selection and workspace capacity, never the admitted key set or numerical result. Inputs,
 * context, output, and live workspace allocations are pairwise non-overlapping.
 */
void context_softmax_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                               const Tensor& context_lengths, const Tensor& valid_columns,
                               const Tensor& table_rows, AttentionHeadGeometry geometry,
                               float scale, const PagedKVBatchLayerView& context,
                               ContextAttentionExecutionEnvelope envelope,
                               WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

/**
 * Return transient capacity for every T in the inclusive optimized interval at the exact batch
 * size. The execution envelope is fixed; invalid profiles or intervals throw.
 */
[[nodiscard]] std::size_t context_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, ContextAttentionExecutionEnvelope envelope,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t batch_size);

} // namespace ninfer::ops
