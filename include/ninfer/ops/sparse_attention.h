#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Qwen Sparse Attention (QSA) of Qwen3.8-Flash-Next (`qwen4_exp`): an indexer selects at most
 * `budget` cached tokens in whole blocks of `ratio` tokens, plus the incomplete tail, and the
 * attention layer runs grouped-query softmax attention over exactly that selection.
 *
 * The three Ops below are the persistent append, the selection and the selected-key attention.
 * Projections, Q/K norms, the attention RoPE and the output gate belong to the caller.
 *
 * Indexer geometry. The indexer has `index_heads` query heads of `index_dim`=128 and one key
 * head. The first `rotary_dim` features of indexer queries and pooled keys rotate with the
 * rotate-half convention: frequency pair i (rotating features i and i+rotary_dim/2) uses
 * angle = FP32(FP32(position) * inv_freq_i) with inv_freq_i = FP32(1 /
 * FP32(theta^(2i/rotary_dim))), the Hugging Face FP32 table and FP32 product, where position is
 * axis `pair_axes[i]` of the token's three multimodal RoPE positions. The FP32 angle is part of the
 * definition: at position 2^16 one FP32 ulp of the angle is ~0.008 rad, far above the other
 * rounding effects. Text tokens carry three equal positions. Qwen3.8-Flash-Next uses interleaved
 * MRoPE sections [11,11,10], i.e. pair_axes[i]=i%3.
 */
inline constexpr std::int32_t kQsaIndexKeyDim = 128;
/** Index record per cached token: 128 raw BF16 key words, three I32 positions, two pad words. */
inline constexpr std::int32_t kQsaIndexRecordWords = 136;
/** Pooled block-key plane: each of a block's `ratio`=4 token slots holds 32 of its 128 words. */
inline constexpr std::int32_t kQsaBlockKeyWords = 32;

struct QsaGeometry {
    std::int32_t query_heads = 0;
    std::int32_t kv_heads    = 0;
    std::int32_t head_dim    = 0;
    std::int32_t index_heads = 0;
    std::int32_t index_dim   = 0;
    std::int32_t budget      = 0; // tokens; budget / ratio whole blocks are kept
    std::int32_t ratio       = 0;
    std::int32_t rotary_dim  = 0;
    float rope_theta         = 0.0F;
    float rms_norm_eps       = 0.0F;
    std::array<std::uint8_t, 64> pair_axes{};

    [[nodiscard]] constexpr std::int32_t block_topk() const noexcept { return budget / ratio; }

    [[nodiscard]] constexpr std::int32_t max_selected() const noexcept {
        return budget + ratio - 1;
    }
};

/** The registered Qwen3.8-Flash-Next geometry: 24/2 heads of 256, indexer 4x128, 2048 in 4s. */
[[nodiscard]] QsaGeometry qwen3_8_flash_next_qsa_geometry();

/**
 * Persistent QSA index planes of one attention layer, paged like the K/V planes (64 tokens per
 * page, page-major). `records` is BF16 [136,64,1,pages]; `block_keys` is BF16 [32,64,1,pages].
 * `block_tables` is I32 [table_stride, rows]; it may be the K/V table itself. Both planes are
 * opaque Op state: only qsa_append writes them and only qsa_select reads them.
 */
struct QsaIndexPlanes {
    Tensor records;
    Tensor block_keys;
    Tensor block_tables;
};

/**
 * Append attention K/V and indexer state for `W` columns of `B` lanes.
 *
 * `k`/`v` are contiguous BF16 [256, kv_heads, W, B] after the attention key norm and RoPE;
 * `index_keys` is contiguous BF16 [128, W, B] raw indexer keys (no norm, no RoPE);
 * `rope_positions` is I32 [W*B, 1|3] axis-major (one column means all three axes are equal);
 * `cache_positions` is I32 [W, B]; `table_rows` I32 [B] selects each lane's block-table row and
 * `valid_columns` (empty or I32 [B]) limits each lane to its first columns. `key_norm` is the
 * BF16 [128] zero-centred indexer key norm weight.
 *
 * K/V storage is byte-identical to kv_cache_append for every cache profile (BF16 K and FP16 V;
 * FP8-row, INT8-G64, NVFP4-G16 and K8V4 with their Hadamard preparation). The raw index key and
 * the token's three positions are stored exactly. After every column of the call is stored, each
 * valid column whose position p completes a block (p % ratio == ratio-1) publishes that block's
 * pooled key
 *
 *   kbar_b = RoPE(offset_rmsnorm(BF16(mean_{j<ratio} raw[ratio*b+j]), key_norm), pos(ratio*b))
 *
 * rounded once to BF16 into `block_keys`, where offset_rmsnorm(x,w) = x/sqrt(mean(x^2)+eps)*(1+w).
 * The pooled mean is rounded to BF16 as upstream does; the RMSNorm and RoPE run in FP32. A block
 * is therefore published exactly when its last token is appended; re-appending any of its tokens
 * (speculative rollback) re-publishes it because tokens are appended in position order.
 * Every addressed row is overwritten and no unrelated row is written. Inputs and every plane are
 * pairwise non-overlapping. The Op owns no allocation, frontier or commit authority.
 */
void qsa_append(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                const Tensor& rope_positions, const Tensor& cache_positions,
                const Tensor& valid_columns, const Tensor& table_rows, const Tensor& key_norm,
                const QsaGeometry& geometry, const PagedKVBatchLayerView& kv,
                const QsaIndexPlanes& index, cudaStream_t stream);

/**
 * Transient capacity of qsa_select for every call of at most `columns` = W*B columns whose
 * visible extents do not exceed `max_visible_keys`.
 */
[[nodiscard]] std::size_t qsa_select_workspace_bytes(const QsaGeometry& geometry,
                                                     std::uint32_t max_visible_keys,
                                                     std::int32_t width, std::int32_t lanes);

/**
 * Select the visible cache indices of every query column.
 *
 * `index_query` is contiguous BF16 [index_heads*128, W, B] raw indexer query rows,
 * `query_rope_positions` is I32 [W*B, 1|3] and `query_norm` the BF16 [128] zero-centred weight.
 * Column (w,b) sees cache indices [0, n) with n = cache_positions[w,b] + 1 and m = floor(n/ratio)
 * complete blocks. With q_h = RoPE(offset_rmsnorm(index_query_h, query_norm), query position) and
 * kbar_b the published pooled key of qsa_append,
 *
 *   score_b = sum_h relu(q_h . kbar_b) / sqrt(128).
 *
 * When m <= block_topk every visible index [0,n) is selected in order. Otherwise the block_topk
 * highest-scoring blocks (the lower block index wins an exact tie) are expanded to their tokens in
 * ascending block order, followed by the incomplete tail [ratio*m, n). `selected` is I32
 * [max_selected, W, B]; `counts` (I32 [W,B]) receives the number of written indices. Columns
 * beyond `valid_columns` produce count zero. Entries past a count are unspecified.
 *
 * The oracle evaluates q_h, kbar_b (from the raw keys, with the BF16 pooled mean) and the scores in
 * FP64; a production selection may differ from it only at near ties of the selection cut, because
 * the published keys are BF16 and the scores accumulate in FP32.
 *
 * `max_visible_keys` is a host launch/workspace promise bounding every live n; it fixes the
 * launch extents, so calls with fixed tensor views stay CUDA Graph update-compatible across
 * envelopes. No device metadata is copied to the host.
 */
void qsa_select(const Tensor& index_query, const Tensor& query_rope_positions,
                const Tensor& cache_positions, const Tensor& valid_columns,
                const Tensor& table_rows, const Tensor& query_norm, const QsaIndexPlanes& index,
                const QsaGeometry& geometry, std::uint32_t max_visible_keys,
                WorkspaceArena& workspace, Tensor& selected, Tensor& counts, cudaStream_t stream);

/**
 * Softmax attention restricted to selected cache indices.
 *
 * `q` is contiguous BF16 [256, query_heads, W, B] after norm and RoPE; query head h reads KV head
 * floor(h / (query_heads/kv_heads)) with at most 16 query heads per KV head. `selected`/`counts`
 * come from qsa_select. For each column and query head the oracle is the scaled dot-product
 * softmax over exactly the selected keys, reading K/V at their storage boundary as the dense
 * causal_softmax_attention contract defines it (FP16 V for the BFloat16 profile; for a quantized
 * profile each stored code times its scale in the stored basis, with q rotated like K and the
 * output of a rotated V rotated back). `out` has the shape of q; a column with count zero writes
 * exact zeros. `workspace` holds qsa_attention_workspace_bytes for the call's W*B columns.
 *
 * Narrow calls split each column's keys over several CTAs, which changes the rounding of the
 * result. `split_columns` (>= W*B; 0 = W*B) makes the call split as a call of that many columns
 * would, so each column's output is bit-identical to that of the wider call it is a part of.
 */
void qsa_attention(const Tensor& q, const Tensor& selected, const Tensor& counts,
                   const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                   std::int32_t split_columns = 0);

/** Workspace of qsa_attention for every call of at most `columns` columns. */
[[nodiscard]] std::size_t qsa_attention_workspace_bytes(const QsaGeometry& geometry,
                                                        std::int32_t columns);

// ---------------------------------------------------------------------------------------------
// QSA K/V staging (active KV in RAM, docs/maintainer/active-kv-in-ram.md)
//
// The paged K/V planes a QSA attention reads may live in pinned, device-mapped Host memory (any
// address the device can dereference). A staging region is device memory laid out as K/V pages
// (the same planes, profile and heads) behind an identity table: the Ops below copy the K/V rows
// a call will read into it and give qsa_attention a selection over the region, so qsa_attention
// reads exactly the bytes, in exactly the order, it would read from the source pages and its
// output is bit-identical. Only K/V code and scale planes are staged; the index planes are not.

/** Per-layer counters a staging call adds to when the caller asks for them (`stats`). */
enum QsaStagingStat : std::int32_t {
    kQsaStagingHits = 0,      // cacheable block whose slot already held it
    kQsaStagingMisses,        // cacheable block copied into its slot
    kQsaStagingShared,        // block another column of the same call had claimed
    kQsaStagingUncacheable,   // block not wholly before the window, copied into the overflow
    kQsaStagingCollisions,    // slot claimed by another block in the same call, overflow copy
    kQsaStagingPages,         // pages copied by qsa_stage_visible_pages
    kQsaStagingBlockCalls,    // qsa_stage_selected_blocks calls
    kQsaStagingPageCalls,     // qsa_stage_visible_pages calls
    kQsaStagingUnionBlocks,   // qsa_count_selected_blocks: distinct selected complete blocks
    kQsaStagingVisibleBlocks, // qsa_count_selected_blocks: complete blocks the last column sees
    kQsaStagingUnionCalls,    // qsa_stage_selected_union calls
    kQsaStagingUnionStaged,   // blocks copied by qsa_stage_selected_union
    kQsaStagingStatCount
};

/** Words of QsaStagingRegion::counters per attention layer. */
inline constexpr std::int32_t kQsaStagingCounterWords = 8;

/**
 * Persistent staging state of one device for `layers` attention layers and the `table_rows` KV
 * table rows a call may name (one per concurrent request). Region slot s (a 4-token block)
 * occupies rows (s % 16) * 4 .. +3 of region page s / 16: layer l's cache slots are
 * [l * cache_blocks, (l + 1) * cache_blocks), shared by every table row; the overflow slots follow
 * all layers. The caller owns the memory and initializes it: tags -1, claims 0, counters 0 except
 * word 0 (epoch) = 1, row_starts 0, stats 0, block_tables[i] = i, table_row = 0.
 */
struct QsaStagingRegion {
    PagedKVBatchLayerView pages; // planes [leading, 64, kv_heads, total_pages], identity table
    Tensor table_row; // I32 [table_rows] = 0: the identity row, one entry per lane of a call
    // I64 [layers * cache_blocks]: (generation << 32 | key) of the block a slot holds, key =
    // table row << 20 | block index (32 bits: table rows below 4096, blocks below 2^20); -1 =
    // empty.
    Tensor tags;
    Tensor claims;   // I64 [layers * cache_blocks]: (epoch << 32 | key) of a call's claimant
    Tensor counters; // I32 [layers * kQsaStagingCounterWords]: epoch, finished CTAs, overflow
                     // cursor, spare, generation, 3 spare words
    // I32 [layers * table_rows]: 1 + the first position of each table row's last decode/verify
    // call of the layer (0: none yet).
    Tensor row_starts;
    Tensor stats; // I64 [layers * kQsaStagingStatCount]
    std::int32_t layers          = 0;
    std::int32_t table_rows      = 1;
    std::int32_t cache_blocks    = 0; // per layer, a multiple of 16
    std::int32_t overflow_blocks = 0; // a multiple of 16

    [[nodiscard]] std::int32_t total_pages() const noexcept {
        return (layers * cache_blocks + overflow_blocks) / 16;
    }
};

/**
 * Copies whole pages for a prefill call of W columns of one lane (cache_positions I32 [W,1],
 * valid_columns empty or I32 [1], table_rows I32 [1]): with n = 1 + the largest cache position of
 * the valid columns, logical page i < ceil(n / 64) of the lane's row of kv.block_tables is copied,
 * every K/V code and scale plane and every head, from `kv` to region page i; then
 * qsa_attention(q, selected, counts, region.table_row[0, 1), region.pages, ...) equals
 * qsa_attention(q, selected, counts, table_rows, kv, ...) bit for bit for any selection of
 * positions below n. Region pages from ceil(n / 64) on are not written. State: every tag of every
 * layer becomes -1 (the copied pages overwrite cache slots). `max_visible_keys` is the host launch
 * envelope (>= n, ceil(max_visible_keys / 64) <= region.total_pages()). With `stats`, the call adds
 * to layer 0's counters (it has no layer).
 */
void qsa_stage_visible_pages(const PagedKVBatchLayerView& kv, const Tensor& cache_positions,
                             const Tensor& valid_columns, const Tensor& table_rows,
                             std::uint32_t max_visible_keys, const QsaStagingRegion& region,
                             bool stats, cudaStream_t stream);

/** 4-token blocks a staging region holds (every layer's cache slots and the overflow). */
[[nodiscard]] inline std::int64_t qsa_staging_region_blocks(const QsaStagingRegion& region) {
    return static_cast<std::int64_t>(region.layers) * region.cache_blocks + region.overflow_blocks;
}

/**
 * Prefill when the region cannot hold one layer of the visible context: copies the 4-token blocks
 * the selections of a call of W columns of one lane name (selected I32 [max_selected, W, 1] and
 * counts I32 [W, 1] as qsa_select writes them, table_rows I32 [1]) into region slots 0, 1, ... in
 * ascending block order, and rewrites the selection onto them: for every column w and entry
 * j < counts[w], rewritten[j, w] = slot * 4 + (selected[j, w] & 3), where the region rows of
 * `slot` hold, when the call completes, the K/V rows (every plane and head) of block
 * selected[j, w] / 4 of the lane's table row. Entries past a count are not written. Then
 * qsa_attention(q, rewritten, counts, region.table_row[0, 1), region.pages, ...) equals
 * qsa_attention(q, selected, counts, table_rows, kv, ...) bit for bit. The distinct blocks must
 * fit qsa_staging_region_blocks(region), which W * ceil(max_selected / 4) <= it guarantees (a
 * larger union, or a block at or beyond ceil(max_visible_keys / 4), traps). `max_visible_keys` is
 * the host launch envelope (>= every cache position + 1). State: every tag of every layer becomes
 * -1. `workspace` holds qsa_stage_selected_union_workspace_bytes(max_visible_keys). With `stats`
 * the call adds to layer 0's counters.
 */
void qsa_stage_selected_union(const Tensor& selected, const Tensor& counts,
                              const Tensor& table_rows, const PagedKVBatchLayerView& kv,
                              std::uint32_t max_visible_keys, const QsaStagingRegion& region,
                              bool stats, WorkspaceArena& workspace, Tensor& rewritten,
                              cudaStream_t stream);

[[nodiscard]] std::size_t qsa_stage_selected_union_workspace_bytes(std::uint32_t max_visible_keys);

/**
 * Copies the selected blocks of a decode/verify call of W columns of each of B lanes into layer
 * `layer`'s part of the region and rewrites the selection onto it. `selected` I32
 * [max_selected, W, B] and `counts` I32 [W, B] follow qsa_select's contract: each aligned group of
 * four entries is one 4-token block (a count above max_selected, a group naming two blocks, or a
 * table row outside [0, region.table_rows), traps: the device context is lost rather than a read
 * past the selection). `cache_positions` I32 [W, B], `table_rows` I32 [B] (distinct rows).
 * `rewritten` (I32 shaped like `selected`, not aliasing it) receives, for every column (w, b) and
 * entry j < counts[w, b],
 *
 *   rewritten[j, w, b] = slot * 4 + (selected[j, w, b] & 3)
 *
 * where the region rows of `slot` hold, when the call completes, the K/V rows (every plane and
 * head) of block selected[j, w, b] / 4 of lane b's table row in `kv`. Entries past a count are not
 * written. Then qsa_attention(q, rewritten, counts, region.table_row[0, B), region.pages, ...)
 * equals qsa_attention(q, selected, counts, table_rows, kv, ...) bit for bit.
 *
 * State: a block of lane b wholly before its window's first position p0(b) = cache_positions[0, b]
 * is cacheable. It maps to slot (block + row * (cache_blocks / 8)) % cache_blocks of the layer
 * (row = table_rows[b]; lanes share the layer's slots) and is copied only when the slot's tag
 * differs from (generation << 32 | key); columns of the call asking for the same block of the
 * same row share its slot; a cacheable block whose slot another block claimed in this call, and
 * every block that is not cacheable, is copied into the overflow
 * (W * B * ceil(max_selected / 4) <= overflow_blocks). The call advances the layer's epoch,
 * records each lane's p0 for its row, and starts a new generation (forgetting every cached block
 * of the layer) when some lane's p0 is below its row's previous call's (the caller may then have
 * rewritten committed rows of the row). Cached bytes are trusted: the caller calls
 * qsa_staging_invalidate whenever K/V bytes of a cacheable block of a table row change or a table
 * row maps another sequence, before the next call. No host metadata: graph-capturable.
 */
void qsa_stage_selected_blocks(const Tensor& selected, const Tensor& counts,
                               const Tensor& cache_positions, const Tensor& table_rows,
                               const PagedKVBatchLayerView& kv, const QsaStagingRegion& region,
                               std::int32_t layer, bool stats, Tensor& rewritten,
                               cudaStream_t stream);

/** Sets every tag of every layer to -1 (enqueued on `stream`). */
void qsa_staging_invalidate(const QsaStagingRegion& region, cudaStream_t stream);

/**
 * Diagnostics: adds to layer `layer`'s stats the number of distinct complete blocks (blocks b
 * with 4b + 3 <= the column's cache position) named by the selections of a call of W columns of
 * one lane, and the number of complete blocks below its largest cache position + 1. `workspace`
 * holds qsa_count_selected_blocks_workspace_bytes(max_visible_keys); max_visible_keys bounds every
 * cache position + 1.
 */
void qsa_count_selected_blocks(const Tensor& selected, const Tensor& counts,
                               const Tensor& cache_positions, std::uint32_t max_visible_keys,
                               const QsaStagingRegion& region, std::int32_t layer,
                               WorkspaceArena& workspace, cudaStream_t stream);

[[nodiscard]] std::size_t qsa_count_selected_blocks_workspace_bytes(std::uint32_t max_visible_keys);

} // namespace ninfer::ops
