#pragma once

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/record_view.h"
#include "core/tensor.h"
#include "ninfer/kv_cache_format.h"
#include "ninfer/ops/attention_geometry.h"
#include "ninfer/ops/context_bucket.h"
#include "ninfer/ops/ragged.h"

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Parameters for relative-attention input preparation.
 * All thresholds and epsilons are finite and positive; when use_log_scaling is true,
 * log_scaling_floor > 0 and log_scaling_alpha >= 0 are finite.
 */
struct RelativeAttentionPrepareParams {
    float qk_clip_threshold        = 22.627416997969522f; // 16 * sqrt(2)
    float rel_clip_threshold       = 8.0f;
    float clip_eps                 = 1e-6f;
    float v_norm_eps               = 1e-6f;
    bool use_log_scaling           = false; // true on Full layers, false on SWA
    std::int32_t log_scaling_floor = 8192;
    float log_scaling_alpha        = 0.1f;
};

/**
 * Input views for relative_attention_prepare.
 * Batched forms use 4D/3D/2D shapes with trailing [W, B]; ragged forms use flattened [N].
 */
struct RelativeAttentionPrepareInputs {
    Tensor q;          // Contiguous BF16 [128, 24, W, B] or [128, 24, N]
    Tensor r;          // Contiguous BF16 [16,  24, W, B] or [16,  24, N]
    Tensor k;          // Contiguous BF16 [128,  8, W, B] or [128,  8, N]
    Tensor v;          // Contiguous BF16 [128,  8, W, B] or [128,  8, N]
    Tensor q_scale;    // Contiguous BF16 [128, 24]
    Tensor r_scale;    // Contiguous BF16 [16,  24]
    // Depthwise width-4 conv weights: contiguous BF16 [1024, 4] (1024 = 128 * 8).
    // Static mode (k_dyn/v_dyn empty): tap k in 0..3 is oldest-to-newest (lag = 3 - k).
    // Dynamic mode (k_dyn/v_dyn non-empty): tap l in 0..3 is newest-to-oldest (lag = l).
    Tensor k_conv_weight;
    Tensor v_conv_weight;
    Tensor k_dyn;      // Optional contiguous BF16 [4, 8, W, B] or [4, 8, N]; empty => static conv
    Tensor v_dyn;      // Optional contiguous BF16 [4, 8, W, B] or [4, 8, N]; matches k_dyn presence
    Tensor positions;  // Contiguous Device I32 [W, B] or [N], sequential nonnegative per sequence
    // Optional fused sink projection (empty => disabled):
    Tensor sink_x;      // Optional contiguous BF16 [H, W, B] or [H, N]
    Tensor sink_weight; // Optional contiguous BF16 [H, 24] (H contiguous, 24 heads)
};

/**
 * Output views for relative_attention_prepare.
 * All non-empty outputs are mutually disjoint and disjoint from all inputs and cache/state storage.
 */
struct RelativeAttentionPrepareOutputs {
    Tensor q_out;    // Contiguous BF16 [128, 24, W, B] or [128, 24, N]
    Tensor r_out;    // Contiguous BF16 [16,  24, W, B] or [16,  24, N]
    Tensor sink_out; // Optional contiguous BF16 [24, W, B] or [24, N] (required iff sink_x non-empty)
};

/**
 * Temporary contiguous current-chunk encoded K/V buffer for SWA prepare -> attention -> commit.
 * Matches KvCacheStorage encoding directly so attention observes the exact persistent codec
 * boundary and post-attention commit is a pure copy without re-quantization:
 *   - KvCacheStorage::BFloat16: k is BF16 [128, 8, ...], v is FP16 [128, 8, ...], scales empty.
 *   - KvCacheStorage::Fp8E4m3fnGroup128Ue8m0: k/v are FP8_E4M3FN [128, 8, ...],
 *     k_scales/v_scales are U8 [1, 8, ...].
 * Trailing shape is [W, B] for batched calls and [N] for ragged calls.
 */
struct RelativeAttentionChunkKV {
    KvCacheStorage storage = KvCacheStorage::BFloat16;
    Tensor k;
    Tensor v;
    Tensor k_scales;
    Tensor v_scales;
};

/**
 * Fused relative-attention input preparation:
 *
 * 1. Q/R per-head scaling, L2 clipping, and optional log-scaling tau(pos):
 *      tau(p)    = use_log_scaling ? (1.0f + alpha * logf(max(1.0f, (p + 1.0f) / floor))) : 1.0f
 *      q_scaled  = FP32(q) * FP32(q_scale)
 *      q_clipped = RN_BF16(q_scaled * min(qk_clip / (||q_scaled||_2 + clip_eps), 1.0f))
 *      q_out     = use_log_scaling ? RN_BF16(FP32(q_clipped) * tau(p)) : q_clipped
 *      (Identical formula for r with r_scale and rel_clip_threshold.)
 *
 * 2. Causal width-4 K/V short convolution using BF16 [1024, 3] history state (oldest to newest):
 *      - Static:  y = RN_BF16((sum_{k=0..3} w[c, k] * u[c, t - 3 + k] + u[c, t]) / sqrt(5))
 *      - Dynamic: y = RN_BF16((sum_{l=0..3} (w[c, l] + dyn[l, head(c), t]) * u[c, t - l]
 *                              + u[c, t]) / sqrt(9))
 *    Note the exact RN_BF16(y) boundary before K L2-clip / V RMSNorm.
 *
 * 3. Post-conv K/V normalization & direct KV encoding (BOTH BF16 and FP8 G128/UE8M0):
 *      - K_final = y_k * min(qk_clip / (||y_k||_2 + clip_eps), 1.0f)
 *      - V_final = y_v * rsqrt((1 / 128) * sum_d y_v[d]^2 + v_norm_eps)
 *    Encoded directly per storage:
 *      - KvCacheStorage::BFloat16: BF16 K + FP16_RNE V.
 *      - KvCacheStorage::Fp8E4m3fnGroup128Ue8m0: E4M3FN codes + 1 U8 UE8M0 scale per 128-dim head
 *        vector (exact codec in ninfer/kv_cache_format.h, no Hadamard transform).
 *    SWA Decode (W == 1 direct): writes slot (pos % 1024) of CyclicKVCacheLayerView directly
 *      (single-token overwrite only displaces dead slot pos - 1024).
 *    SWA Multi-token Direct / Snapshot / Ragged: writes contiguous encoded K/V into
 *      RelativeAttentionChunkKV so committed cyclic slots needed by earlier queries in the chunk
 *      remain untouched until relative_softmax_attention finishes (and accepted tokens are
 *      committed via relative_attention_commit_cyclic_kv).
 *    Full: writes (page = pos / 64, slot = pos % 64) of PagedKVBatchLayerView directly (D=128,
 *      Hkv=8); every position has a unique page/slot so no live history is ever displaced.
 *
 * 4. Optional fused sink_out[h, t] = RN_BF16(dot(sink_weight[:, h], sink_x[:, t])) for any H > 0.
 *
 * Batched forms accept empty valid_columns (all W live) or Device I32 [B] with 0 <= V_b <= W.
 * Inactive columns j >= V_b write exact BF16 zero to q_out/r_out/sink_out and do not mutate
 * conv state, chunk_kv, or KV cache. V_b == 0 reads no state/input/position/offset metadata for
 * row b. Exact per-row alias k_state_out == k_state_in (and v_state_out == v_state_in) is allowed.
 * Snapshot forms write post-column-j width-3 windows to snapshots[:, :, snapshot_base_slots[b] + j]
 * for j < V_b. Zero caller workspace; CUDA Graph capture/replay safe.
 */
void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const Tensor& valid_columns, ConstStateFieldView k_state_in,
                                StateFieldView k_state_out, ConstStateFieldView v_state_in,
                                StateFieldView v_state_out, CyclicKVCacheLayerView kv_cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const Tensor& valid_columns, ConstStateFieldView k_state_in,
                                StateFieldView k_state_out, ConstStateFieldView v_state_in,
                                StateFieldView v_state_out, RelativeAttentionChunkKV chunk_kv,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

void relative_attention_prepare_snapshot(const RelativeAttentionPrepareInputs& inputs,
                                         const Tensor& valid_columns,
                                         const Tensor& snapshot_base_slots,
                                         ConstStateFieldView k_state_in, Tensor& k_snapshots,
                                         ConstStateFieldView v_state_in, Tensor& v_snapshots,
                                         RelativeAttentionChunkKV chunk_kv,
                                         const RelativeAttentionPrepareParams& params,
                                         RelativeAttentionPrepareOutputs outputs,
                                         CudaExecutionView execution);

void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const Tensor& valid_columns, const Tensor& kv_table_rows,
                                ConstStateFieldView k_state_in, StateFieldView k_state_out,
                                ConstStateFieldView v_state_in, StateFieldView v_state_out,
                                PagedKVBatchLayerView kv_cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

void relative_attention_prepare_snapshot(const RelativeAttentionPrepareInputs& inputs,
                                         const Tensor& valid_columns, const Tensor& kv_table_rows,
                                         const Tensor& snapshot_base_slots,
                                         ConstStateFieldView k_state_in, Tensor& k_snapshots,
                                         ConstStateFieldView v_state_in, Tensor& v_snapshots,
                                         PagedKVBatchLayerView kv_cache,
                                         const RelativeAttentionPrepareParams& params,
                                         RelativeAttentionPrepareOutputs outputs,
                                         CudaExecutionView execution);

void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const RaggedSequenceView& ragged, ConstStateFieldView k_state_in,
                                StateFieldView k_state_out, ConstStateFieldView v_state_in,
                                StateFieldView v_state_out, RelativeAttentionChunkKV chunk_kv,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

void relative_attention_prepare(const RelativeAttentionPrepareInputs& inputs,
                                const RaggedSequenceView& ragged, const Tensor& kv_table_rows,
                                ConstStateFieldView k_state_in, StateFieldView k_state_out,
                                ConstStateFieldView v_state_in, StateFieldView v_state_out,
                                PagedKVBatchLayerView kv_cache,
                                const RelativeAttentionPrepareParams& params,
                                RelativeAttentionPrepareOutputs outputs,
                                CudaExecutionView execution);

/**
 * Commit already-encoded SWA chunk K/V into CyclicKVCacheLayerView (slot = pos % 1024).
 *
 * Batched form: commits prefix [0, counts[b]) (or all W if counts is empty) for each row b.
 * If counts[b] > 1024, only the surviving tail [counts[b] - 1024, counts[b]) is written.
 * Zero count reads no positions, offsets, or chunk_kv for row b and mutates no cache bytes.
 *
 * Ragged form: commits each segment's surviving tail (last min(seq_lengths[s], 1024) tokens)
 * from packed chunk_kv. Zero caller workspace; CUDA Graph capture/replay safe.
 */
void relative_attention_commit_cyclic_kv(const RelativeAttentionChunkKV& chunk_kv,
                                         const Tensor& positions, const Tensor& counts,
                                         CyclicKVCacheLayerView kv_cache,
                                         CudaExecutionView execution);

void relative_attention_commit_cyclic_kv(const RelativeAttentionChunkKV& chunk_kv,
                                         const RaggedSequenceView& ragged,
                                         const Tensor& positions, CyclicKVCacheLayerView kv_cache,
                                         CudaExecutionView execution);

/**
 * Query-side operands for relative_softmax_attention.
 * Public API passes compact prepared `r` and `rel_logits_proj` [16, 1024] (32 KB),
 * never a pre-materialized sheared bias tensor.
 */
struct RelativeAttentionQueryInputs {
    Tensor q;               // Contiguous BF16 [128, 24, W, B] or [128, 24, N]
    Tensor r;               // Contiguous BF16 [16,  24, W, B] or [16,  24, N]
    Tensor rel_logits_proj; // Contiguous BF16 [16, 1024] (d_rel=16 fastest, distance 0..1023 outer)
    Tensor gate;            // Contiguous BF16 [128, 24, W, B] or [128, 24, N]
    Tensor sink;            // Contiguous BF16 [24,  W, B] or [24, N]
    Tensor positions;       // Contiguous Device I32 [W, B] or [N]
};

/**
 * Causal Grouped-Query Attention with Relative Logit Bias + Fused Gate/Sink Epilogue.
 *
 * Supported profile:
 *   geometry = {head_dim = 128, query_heads = 24, kv_heads = 8} (G = 3),
 *   scale = 1 / sqrt(128), d_rel = 16, rel_extent = 1024, SWA window = 1024,
 *   storage in {KvCacheStorage::BFloat16, KvCacheStorage::Fp8E4m3fnGroup128Ue8m0},
 *   buckets K2 through M1.
 *
 * Exact math for each live query (h, i) at absolute position p_i (kvh = h / 3):
 *   - Visible key set J:
 *       SWA (CyclicKVCacheLayerView, capacity=1024): J = [max(0, p_i - 1023), p_i].
 *         - Single-token decode fast path (batched W == 1 with empty chunk_kv): all keys in J
 *           read cache at slot p_j % 1024 (including the newly written p_i).
 *         - Two-segment path (chunk_kv non-empty; required for batched W > 1 and all ragged):
 *           let L be the sequence's chunk start position (positions[0, b] or
 *           positions[cu_seqlens[s]]). Committed history keys p_j < L read cache[p_j % 1024];
 *           current-chunk keys p_j >= L read chunk_kv at local index (p_j - L).
 *       Full (PagedKVBatchLayerView):                J = [0, p_i], page = p_j / 64, slot = p_j % 64
 *   - Decoded K[:, kvh, p_j], V[:, kvh, p_j] follow cache.storage (BF16/FP16 or FP8 G128/UE8M0).
 *   - Distance d = p_i - p_j >= 0:
 *       rel_bias(p_j) = (d < 1024) ? sum_{u=0..15} FP32(r[u, h, i]) * FP32(rel_logits_proj[u, d])
 *                                  : 0.0f
 *   - score(p_j)   = scale * dot(q[:, h, i], K[:, kvh, p_j]) + rel_bias(p_j)
 *   - lse          = log(sum_{j in J} exp(score(p_j)))                           (natural log)
 *   - attn_val     = sum_{j in J} softmax(score)[j] * V[:, kvh, p_j]
 *   - out[:, h, i] = RN_BF16(attn_val * sigmoid(FP32(gate[:, h, i]))
 *                                     * sigmoid(lse - FP32(sink[h, i])))
 *
 * Inactive batched columns (i >= valid_columns[b]) write exact BF16 zero and read no cache.
 * Cache and chunk_kv are strictly read-only; out is disjoint from all inputs, cache, chunk_kv,
 * and workspace.
 */
void relative_softmax_attention(const RelativeAttentionQueryInputs& inputs,
                                const Tensor& valid_columns, AttentionHeadGeometry geometry,
                                std::uint32_t window, float scale,
                                const ConstCyclicKVCacheLayerView& cache,
                                const RelativeAttentionChunkKV& chunk_kv, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

void relative_softmax_attention(const RelativeAttentionQueryInputs& inputs,
                                const Tensor& valid_columns, const Tensor& kv_table_rows,
                                AttentionHeadGeometry geometry, float scale,
                                const ConstPagedKVBatchLayerView& cache, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

void relative_softmax_attention(const RelativeAttentionQueryInputs& inputs,
                                const RaggedSequenceView& ragged, AttentionHeadGeometry geometry,
                                std::uint32_t window, float scale,
                                const ConstCyclicKVCacheLayerView& cache,
                                const RelativeAttentionChunkKV& chunk_kv, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

void relative_softmax_attention(const RelativeAttentionQueryInputs& inputs,
                                const RaggedSequenceView& ragged, const Tensor& kv_table_rows,
                                AttentionHeadGeometry geometry, float scale,
                                const ConstPagedKVBatchLayerView& cache, ContextBucket bucket,
                                MemoryArena& workspace, Tensor& out, CudaExecutionView execution);

[[nodiscard]] std::size_t relative_softmax_attention_swa_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, std::uint32_t window,
    ContextBucketRange buckets, std::int32_t batch_size, std::int32_t min_tokens,
    std::int32_t max_tokens, std::int32_t multiprocessor_count);

[[nodiscard]] std::size_t relative_softmax_attention_full_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, ContextBucketRange buckets,
    std::int32_t batch_size, std::int32_t min_tokens, std::int32_t max_tokens,
    std::int32_t multiprocessor_count);

[[nodiscard]] std::size_t relative_softmax_attention_ragged_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage storage, bool is_swa, ContextBucketRange buckets,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t min_segments,
    std::int32_t max_segments, std::int32_t multiprocessor_count);

[[nodiscard]] OpGraphProfile relative_softmax_attention_swa_graph_profile(
    AttentionHeadGeometry geometry, KvCacheStorage storage, std::uint32_t window,
    ContextBucket bucket, std::int32_t batch_size, std::int32_t tokens,
    std::int32_t multiprocessor_count, bool masked = true);

[[nodiscard]] OpGraphProfile relative_softmax_attention_full_graph_profile(
    AttentionHeadGeometry geometry, KvCacheStorage storage, ContextBucket bucket,
    std::int32_t batch_size, std::int32_t tokens, std::int32_t multiprocessor_count,
    bool masked = true);

} // namespace ninfer::ops
