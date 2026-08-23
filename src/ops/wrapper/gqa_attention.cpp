// ninfer::ops - GQA A1/A2/A3 validation and finite route dispatch.
#include "ninfer/ops/gqa_attention.h"

#include "core/layout.h"
#include "ops/launcher/gqa_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim                      = 256;
constexpr std::int32_t kQuantGroup                   = 64;
constexpr float kExpectedScale                       = 0.0625f;
// 6 is the worst value in its neighbourhood on the Volta tensor-core prefill path. That kernel
// packs Br=32 rows per row-tile pass, and rows = TokenTile * GQA group size (6 for this model's
// 24q/4kv geometry), so TokenTile=6 needs 36 rows -> 2 passes, and each pass re-walks the whole
// key range. TokenTile=5 fits 30 rows in a single pass, so the per-chunk cost factor
// (passes/TokenTile) drops from 0.333 to 0.200 -- 1.67x less attention work for the same prompt,
// despite the slightly smaller chunk. Attention measured 71.6% of a 12K prefill, so this is the
// dominant term at long context. See the V100 performance summary.
constexpr std::int32_t kSmallTChunkTokens            = 5;
constexpr std::int32_t kMaximumVerifyTokens          = 16;
constexpr std::int32_t kMaximumBatchSize             = 8;
constexpr std::uint32_t kTwoChunkPromptVisibleKeys   = 512;
constexpr std::uint32_t kThreeChunkPromptVisibleKeys = 1024;

std::int32_t kv_heads_for_q_heads(std::int32_t q_heads, const char* op) {
    if (q_heads == 24) { return 4; }
    if (q_heads == 16) { return 2; }
    throw std::invalid_argument(std::string(op) + ": unsupported Q/KV head geometry");
}

void require_kv_heads(std::int32_t kv_heads, const char* op) {
    if (kv_heads != 4 && kv_heads != 2) {
        throw std::invalid_argument(std::string(op) + ": unsupported KV head geometry");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::uint32_t validate_cache(const PagedKVLayerView& cache, std::int32_t kv_heads, const char* op) {
    if ((cache.dtype != DType::BF16 && cache.dtype != DType::I8) ||
        cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (cache.dtype == DType::BF16 && cache.quant_group != 0) {
        throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have quant_group");
    }
    if (cache.dtype == DType::I8 && cache.quant_group != kQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KV cache must use quant_group 64");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    const DType code_dtype = cache.dtype == DType::I8 ? DType::I8 : DType::BF16;
    if (cache.k_pages.dtype != code_dtype || cache.v_pages.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, op, "block table");
    require_contiguous_nonnull(cache.block_table, op, "block table");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    constexpr std::int32_t groups = kHeadDim / kQuantGroup;
    if (cache.k_scale_pages.dtype != DType::FP16 || cache.v_scale_pages.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k scale pages");
    require_shape(cache.v_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v scale pages");
    require_contiguous_nonnull(cache.k_scale_pages, op, "cache k scale pages");
    require_contiguous_nonnull(cache.v_scale_pages, op, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t validate_batch_cache(const PagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                   const char* op) {
    if ((cache.dtype != DType::BF16 && cache.dtype != DType::I8) ||
        cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (cache.dtype == DType::BF16 && cache.quant_group != 0) {
        throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have quant_group");
    }
    if (cache.dtype == DType::I8 && cache.quant_group != kQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KV cache must use quant_group 64");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_tables.ne[0];
    const std::int32_t table_rows     = cache.block_tables.ne[1];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 || table_rows <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    const DType code_dtype = cache.dtype == DType::I8 ? DType::I8 : DType::BF16;
    if (cache.k_pages.dtype != code_dtype || cache.v_pages.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block tables must be I32");
    }
    require_shape(cache.block_tables, logical_pages, table_rows, 1, 1, op, "block tables");
    require_contiguous_nonnull(cache.block_tables, op, "block tables");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    constexpr std::int32_t groups = kHeadDim / kQuantGroup;
    if (cache.k_scale_pages.dtype != DType::FP16 || cache.v_scale_pages.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k scale pages");
    require_shape(cache.v_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v scale pages");
    require_contiguous_nonnull(cache.k_scale_pages, op, "cache k scale pages");
    require_contiguous_nonnull(cache.v_scale_pages, op, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

void validate_envelope(GqaExecutionEnvelope envelope, const PagedKVLayerView& cache,
                       std::int32_t tokens, const char* op) {
    const std::uint32_t capacity = validate_cache(cache, cache.num_kv_heads, op);
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                                float scale, const char* op) {
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, op);
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    validate_envelope(envelope, cache, tokens, op);
}

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const PagedKVBatchLayerView& cache,
                                        GqaExecutionEnvelope envelope, float scale,
                                        const char* op) {
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, op);
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatchSize ||
        (batch > 1 && width > kMaximumVerifyTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    require_shape(q, kHeadDim, q_heads, width, batch, op, "q");
    require_shape(positions, width, batch, 1, 1, op, "positions");
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns"); }
    require_shape(kv_table_rows, batch, 1, 1, 1, op, "KV table rows");
    require_shape(out, kHeadDim, q_heads, width, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    if (masked) { require_contiguous_nonnull(valid_columns, op, "valid columns"); }
    require_contiguous_nonnull(kv_table_rows, op, "KV table rows");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op);
    if (cache.block_tables.ne[1] < batch || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(width)) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope or table");
    }
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t q_heads,
                                           std::int32_t tokens, std::int32_t splits,
                                           std::int32_t batch_size = 1) {
    return {
        workspace.alloc(DType::BF16, {kHeadDim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
    };
}

#ifdef NINFER_VOLTA_BUILD
// Whether a call with this static profile could ever take the VoltaFlash route.
//
// Deliberately excludes the envelope-shape test that gqa_attention_resolve_route
// applies. The workspace contract is computed from a *planning* envelope
// ({1, capacity}), while the prefill call that actually routes here passes
// {visible, visible}: testing min == max in the capacity probe would report zero
// bytes for a route that then allocates hundreds of megabytes at runtime. The
// capacity probe therefore uses only the envelope's bound, never its shape.
bool volta_flash_route_possible(std::int32_t q_heads, std::int32_t width,
                                std::int32_t batch_size, DType cache_dtype) {
    // 24q/4kv only (ratio 6 -> ncols2 2). 16q/2kv is also instantiated (ratio 8 -> ncols2 8),
    // but it corrupts shared memory: the combine buffer is sized on the host from
    // get_cols_per_warp(), which is 32 on Volta, while the kernel indexes it by the tile
    // type's T_B_KQ::I, and those two disagree once ncols2 is 8. compute-sanitizer reports
    // an invalid 8-byte __shared__ write at fattn-mma-f16.cuh:1483 (the combine-meta store)
    // for every tile. ChunkedSmallT is the correct fallback -- slower, but it is the path
    // 16q/2kv prefill used before the flash route existed. Re-enabling this geometry means
    // reconciling the two cols_per_warp definitions upstream first.
    return (q_heads == 24 || q_heads == 16) && batch_size == 1 &&
           (cache_dtype == DType::BF16 || cache_dtype == DType::I8) &&
           width >= detail::kVoltaFlashMinimumWidth;
}

struct VoltaFlashWorkspace {
    Tensor k_gathered;
    Tensor v_gathered;
    Tensor mask;
    Tensor q_f32;
    Tensor out_f32;
    Tensor dst_meta;
};

// Staging for the Volta flash route. The gathered K/V is sized by the whole
// visible key range (one gather serves every Q-block of a layer); everything else
// is sized by one Q-block, which is why the Q-block width exists at all.
template <class Allocator>
VoltaFlashWorkspace allocate_volta_flash_workspace(Allocator& workspace, std::int32_t q_heads,
                                                   std::int32_t width,
                                                   GqaExecutionEnvelope envelope) {
    const std::int32_t kv_heads = q_heads == 24 ? 4 : 2;
    // Both the gathered K/V and the mask are sized by the padded key extent; see
    // the FATTN_KQ_STRIDE note in gqa_attention_volta_flash.cu.
    const auto visible          = static_cast<std::int32_t>(envelope.max_visible_keys);
    const std::int32_t n_kv =
        ((visible + detail::kVoltaFlashKeyPad - 1) / detail::kVoltaFlashKeyPad) *
        detail::kVoltaFlashKeyPad;
    const std::int32_t tokens   = std::min(width, detail::kVoltaFlashQBlockTokens);

    return {
        workspace.alloc(DType::FP16, {kHeadDim, kv_heads, n_kv, 1}),
        workspace.alloc(DType::FP16, {kHeadDim, kv_heads, n_kv, 1}),
        workspace.alloc(DType::FP16, {n_kv, tokens + detail::kVoltaFlashMaskRowPad, 1, 1}),
        workspace.alloc(DType::FP32, {kHeadDim, q_heads, tokens, 1}),
        workspace.alloc(DType::FP32, {kHeadDim, q_heads, tokens, 1}),
        workspace.alloc(DType::FP32,
                        {2, static_cast<std::int32_t>(
                                detail::gqa_attention_volta_flash_meta_elements(q_heads, tokens)),
                         1, 1}),
    };
}
#endif // NINFER_VOLTA_BUILD

template <typename Launch>
void for_each_small_t_chunk(const Tensor& q, const Tensor& positions, WorkspaceArena& workspace,
                            DType cache_dtype, GqaExecutionEnvelope envelope, Tensor& out,
                            Launch&& launch) {
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += kSmallTChunkTokens) {
        const std::int32_t count = std::min(kSmallTChunkTokens, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], count, cache_dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], count, splits);
        Tensor q_chunk          = q.slice(2, begin, count);
        Tensor position_chunk   = positions.slice(0, begin, count);
        Tensor out_chunk        = out.slice(2, begin, count);
        launch(begin, count, q_chunk, position_chunk, partial, out_chunk);
    }
}

void launch_chunked_small_t(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid_columns,
                            const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                            GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                            cudaStream_t stream) {
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += kSmallTChunkTokens) {
        const std::int32_t count = std::min(kSmallTChunkTokens, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], count, cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], count, splits, q.ne[3]);
        detail::gqa_attention_small_t_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, envelope, begin, count, partial.acc, partial.m,
                                             partial.l, out, stream);
    }
}

void launch_cached_chunked_small_t(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.dtype, envelope, out,
        [&](std::int32_t, std::int32_t, const Tensor& q_chunk, const Tensor& position_chunk,
            SmallTWorkspace& partial, Tensor& out_chunk) {
            detail::gqa_attention_cached_small_t_launch(q_chunk, position_chunk, scale, cache,
                                                        envelope, partial.acc, partial.m, partial.l,
                                                        out_chunk, stream);
        });
}

} // namespace

namespace detail {

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                              std::int32_t batch_size, DType cache_dtype,
                                              GqaExecutionEnvelope envelope) {
    if (width >= 1 && width <= kSmallTChunkTokens) { return GqaAttentionRoute::SmallT; }
    if (batch_size > 1) { return GqaAttentionRoute::ChunkedSmallT; }
#ifdef NINFER_VOLTA_BUILD
    // Volta tiled flash attention, for single-batch prefill wide enough to pay for
    // its staging. ChunkedSmallT stays reachable for everything else -- decode, MTP
    // verify, short prompts -- so a regression here can be bisected by routing alone.
    //
    // Restrictions, each a real precondition rather than caution:
    //  - q_heads == 24 is the only geometry whose gqa_ratio (6) selects ncols2 = 2,
    //    the configuration validated in the V100 implementation. 16q/2kv has ratio 8 and
    //    would select a different instantiation that is not built here.
    //  - min == max visible keys is what prefill passes (chunk_envelope{visible,
    //    visible}); it makes the key extent an exact host-side scalar, which the
    //    kernel needs as a launch parameter. Anything else is not a prefill call.
    //  - the envelope must cover the width, or `base` below would go negative.
    //  - the gather stages BF16 cache rows into FP16, or dequantizes INT8-G64
    //    cache rows into the same FP16 boundary once per layer.
    if (volta_flash_route_possible(q_heads, width, batch_size, cache_dtype) &&
        envelope.min_visible_keys == envelope.max_visible_keys &&
        envelope.max_visible_keys >= static_cast<std::uint32_t>(width)) {
        return GqaAttentionRoute::VoltaFlash;
    }
    // GqaAttentionRoute::Prompt (gqa_attention_prompt_{launch,attention_launch} ->
    // ops/kernel/gqa_attention_prefill_{bf16,i8}.cuh) is a tensor-core flash-attention kernel
    // (ldmatrix/mma.m16n8k16, sm_80+) with no SIMT sibling. ChunkedSmallT instead drives the
    // already-ported/validated small-T decode kernel (TokenTile<=6, intra-chunk causal mask)
    // across the whole width in kSmallTChunkTokens-sized chunks, appending KV per chunk -- the
    // same mechanism MTP verify already relies on for batch_size>1, applied here to ordinary
    // single-batch prefill too. Per-chunk workspace is bounded by kSmallTChunkTokens regardless
    // of total width, so gqa_attention_workspace_capacity_bytes's existing precomputation (which
    // only samples widths up to kMaximumVerifyTokens) already covers this correctly. Slower than
    // tiled flash-attention, but correct -- see the V100 performance summary.
    (void)q_heads;
    (void)envelope;
    return GqaAttentionRoute::ChunkedSmallT;
#else
    (void)cache_dtype;
    const std::uint32_t prompt_visible_keys =
        width <= 2 * kSmallTChunkTokens ? kTwoChunkPromptVisibleKeys : kThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && width <= kMaximumVerifyTokens &&
        envelope.max_visible_keys > prompt_visible_keys) {
        return GqaAttentionRoute::ChunkedSmallT;
    }
    return GqaAttentionRoute::Prompt;
#endif
}

const char* gqa_attention_route_name(GqaAttentionRoute route) {
    switch (route) {
    case GqaAttentionRoute::SmallT:
        return "small_t";
    case GqaAttentionRoute::ChunkedSmallT:
        return "chunked_small_t";
    case GqaAttentionRoute::Prompt:
        return "prompt";
    case GqaAttentionRoute::VoltaFlash:
        return "volta_flash";
    }
    return "unknown";
}

} // namespace detail

std::size_t gqa_attention_workspace_capacity_bytes(std::int32_t q_heads, DType cache_dtype,
                                                   GqaExecutionEnvelope envelope,
                                                   std::int32_t batch_size, std::int32_t min_width,
                                                   std::int32_t max_width) {
    (void)kv_heads_for_q_heads(q_heads, "gqa_attention workspace");
    if ((cache_dtype != DType::BF16 && cache_dtype != DType::I8) || batch_size <= 0 ||
        batch_size > kMaximumBatchSize || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumVerifyTokens) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(max_width)) {
        throw std::invalid_argument("gqa_attention workspace: invalid profile or interval");
    }

    const auto chunk_capacity = [&](std::int32_t width) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q_heads, width, cache_dtype, envelope);
        WorkspaceLayoutBuilder layout;
        (void)allocate_small_t_workspace(layout, q_heads, width, splits, batch_size);
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t width) {
        const detail::GqaAttentionRoute route =
            detail::gqa_attention_resolve_route(q_heads, width, batch_size, cache_dtype, envelope);
        if (route == detail::GqaAttentionRoute::Prompt) { return std::size_t{0}; }
        // VoltaFlash carries its own staging, declared once from max_width above; reporting the
        // chunked cost here instead would over-declare for an interval that routes there.
        if (route == detail::GqaAttentionRoute::VoltaFlash) { return std::size_t{0}; }
        if (route == detail::GqaAttentionRoute::SmallT) { return chunk_capacity(width); }
        std::size_t maximum = 0;
        for (std::int32_t begin = 0; begin < width; begin += kSmallTChunkTokens) {
            maximum =
                std::max(maximum, chunk_capacity(std::min(kSmallTChunkTokens, width - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
#ifdef NINFER_VOLTA_BUILD
    // The VoltaFlash staging scales with the width and the visible key range, so
    // unlike every chunked route its need is NOT bounded by a small fixed chunk and
    // the sampling loop below would miss it entirely. Declare it for any width in
    // the interval that would actually route there.
    // The staging grows monotonically with width (only through
    // min(width, kVoltaFlashQBlockTokens)), so the widest width in the interval
    // bounds every narrower one that also routes here.
    if (volta_flash_route_possible(q_heads, max_width, batch_size, cache_dtype)) {
        WorkspaceLayoutBuilder layout;
        (void)allocate_volta_flash_workspace(layout, q_heads, max_width, envelope);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
#endif // NINFER_VOLTA_BUILD
    if (min_width <= kMaximumVerifyTokens) {
        const std::int32_t last = std::min(max_width, kMaximumVerifyTokens);
        for (std::int32_t width = min_width; width <= last; ++width) {
            maximum = std::max(maximum, exact_capacity(width));
        }
    }
    // Widths above the verify limit are single-batch, so they route to ChunkedSmallT (or, on
    // Volta, to VoltaFlash, declared above). An interval that starts above the limit -- an
    // exact-width declaration for one wide call -- would otherwise sample nothing and declare
    // zero, and the chunk loop would then run off the end of the arena.
    //
    // Such a width costs the widest of its chunks, and its chunk decomposition is
    // {kSmallTChunkTokens} plus the remainder, so the cost repeats with period
    // kSmallTChunkTokens. Sampling one period from the bottom of the tail therefore visits
    // every remainder the interval can produce and stays exact -- which matters, because the
    // callers assert the declaration equals the execution high-water rather than merely
    // bounding it. Sampling a single representative width would not do: chunk cost is not
    // monotonic in the chunk width (the INT8 T=5 split policy is deliberately finer than T=4's).
    if (max_width > kMaximumVerifyTokens) {
        const std::int32_t begin = std::max(min_width, kMaximumVerifyTokens + 1);
        const std::int32_t last  = std::min(max_width, begin + kSmallTChunkTokens - 1);
        for (std::int32_t width = begin; width <= last; ++width) {
            maximum = std::max(maximum, exact_capacity(width));
        }
    }
    return maximum;
}

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "gqa_attention";
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q.ne[1], op);
    require_shape(k, kHeadDim, kv_heads, width, batch, op, "k");
    require_shape(v, kHeadDim, kv_heads, width, batch, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");

    auto scope = workspace.scope();
    const detail::GqaAttentionRoute route =
        detail::gqa_attention_resolve_route(q.ne[1], width, batch, cache.dtype, envelope);
#ifdef NINFER_VOLTA_BUILD
    if (route == detail::GqaAttentionRoute::VoltaFlash) {
        VoltaFlashWorkspace staging =
            allocate_volta_flash_workspace(workspace, q.ne[1], width, envelope);
        detail::gqa_attention_volta_flash_launch(
            q, k, v, positions, kv_table_rows, scale, cache, envelope,
            detail::kVoltaFlashQBlockTokens, staging.k_gathered, staging.v_gathered, staging.mask,
            staging.q_f32, staging.out_f32, staging.dst_meta, out, stream);
        return;
    }
#endif // NINFER_VOLTA_BUILD
    if (route == detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_chunked_small_t(q, k, v, positions, valid_columns, kv_table_rows, scale, cache,
                               envelope, workspace, out, stream);
        return;
    }
    if (route == detail::GqaAttentionRoute::SmallT) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], width, cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], width, splits, batch);
        detail::gqa_attention_small_t_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                             scale, cache, envelope, 0, width, partial.acc,
                                             partial.m, partial.l, out, stream);
        return;
    }
    detail::gqa_attention_prompt_launch(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                        cache, out, stream);
}

void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   PagedKVLayerView cache, cudaStream_t stream) {
    constexpr const char* op = "gqa_kv_append";
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_kv_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_kv_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    require_kv_heads(kv_heads, op);
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_kv_append: T must be positive"); }
    require_shape(k, kHeadDim, kv_heads, tokens, 1, op, "k");
    require_shape(v, kHeadDim, kv_heads, tokens, 1, op, "v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    const std::uint32_t capacity = validate_cache(cache, kv_heads, op);
    if (static_cast<std::uint32_t>(tokens) > capacity) {
        throw std::invalid_argument("gqa_kv_append: T exceeds KV cache capacity");
    }
    detail::gqa_kv_append_launch(k, v, positions, cache, stream);
}

void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "gqa_attention_cached";
    validate_attention_tensors(q, positions, out, cache, envelope, scale, op);

    auto scope = workspace.scope();
    const detail::GqaAttentionRoute route =
        detail::gqa_attention_resolve_route(q.ne[1], q.ne[2], 1, cache.dtype, envelope);
    // A3 accepts no new K/V, so the VoltaFlash launcher's append step has nothing to
    // do and its staging is not allocated here. The chunked route is the cached-path
    // equivalent. Without this, a wide A3 call would fall through to
    // gqa_attention_prompt_attention_launch, which traps below sm_80.
    if (route == detail::GqaAttentionRoute::ChunkedSmallT ||
        route == detail::GqaAttentionRoute::VoltaFlash) {
        launch_cached_chunked_small_t(q, positions, scale, cache, envelope, workspace, out, stream);
        return;
    }
    if (detail::gqa_attention_uses_small_t(q.ne[2])) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], q.ne[2], cache.dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], q.ne[2], splits);
        detail::gqa_attention_cached_small_t_launch(q, positions, scale, cache, envelope,
                                                    partial.acc, partial.m, partial.l, out, stream);
        return;
    }
    detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops
