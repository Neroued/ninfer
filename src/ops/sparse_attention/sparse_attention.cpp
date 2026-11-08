// ninfer::ops - Qwen Sparse Attention wrappers: validate the public contract, then launch.
#include "ninfer/ops/sparse_attention.h"

#include "core/paged_kv_storage.h"
#include "ops/sparse_attention/launch.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(bool condition, const char* op, const std::string& message) {
    if (!condition) { throw std::invalid_argument(std::string(op) + ": " + message); }
}

void require_dense(const Tensor& t, DType dtype, const char* op, const char* label) {
    require(t.dtype == dtype && t.data != nullptr && t.is_contiguous() && t.numel() > 0, op,
            std::string(label) + " must be contiguous and non-empty with the expected dtype");
}

void require_optional_lanes(const Tensor& t, std::int64_t lanes, const char* op,
                            const char* label) {
    if (t.data == nullptr) { return; }
    require_dense(t, DType::I32, op, label);
    require(t.numel() == lanes, op, std::string(label) + " must be I32 [B]");
}

void require_storage(const PagedKVBatchLayerView& kv, const char* op) {
    require(kv.head_dim == 256 && kv.num_kv_heads > 0 && kv.num_kv_heads <= 8, op,
            "KV geometry must be head_dim 256 with at most 8 KV heads");
    const PagedKVStorageLayout layout = paged_kv_storage_layout(kv.storage, kv.head_dim);
    const auto plane                  = [&](const Tensor& data, const Tensor& scales,
                           const PagedKVVectorLayout& vector, const char* label) {
        require(data.data != nullptr && data.dtype == vector.data_dtype &&
                                     data.ne[0] == vector.data_leading_extent && data.ne[1] == kPagedKVPageSize &&
                                     data.ne[2] == kv.num_kv_heads,
                                 op, std::string(label) + " pages do not match the KV-cache profile");
        require(!vector.has_scale() ||
                                     (scales.data != nullptr && scales.dtype == vector.scale_dtype &&
                     scales.ne[0] == vector.scale_leading_extent),
                                 op, std::string(label) + " scales do not match the KV-cache profile");
    };
    plane(kv.k_pages, kv.k_scale_pages, layout.key, "K");
    plane(kv.v_pages, kv.v_scale_pages, layout.value, "V");
    require(kv.block_tables.dtype == DType::I32 && kv.block_tables.data != nullptr, op,
            "block tables must be I32");
}

void require_index(const QsaIndexPlanes& index, const char* op) {
    require(index.records.dtype == DType::BF16 && index.records.data != nullptr &&
                index.records.ne[0] == kQsaIndexRecordWords &&
                index.records.ne[1] == kPagedKVPageSize && index.records.ne[2] == 1,
            op, "index records must be BF16 [136,64,1,pages]");
    require(index.block_keys.dtype == DType::BF16 && index.block_keys.data != nullptr &&
                index.block_keys.ne[0] == kQsaBlockKeyWords &&
                index.block_keys.ne[1] == kPagedKVPageSize && index.block_keys.ne[2] == 1 &&
                index.block_keys.ne[3] == index.records.ne[3],
            op, "block keys must be BF16 [32,64,1,pages] over the record pages");
    require(index.block_tables.dtype == DType::I32 && index.block_tables.data != nullptr, op,
            "index block tables must be I32");
}

void require_positions(const Tensor& rope_positions, std::int64_t tokens, const char* op) {
    require(rope_positions.dtype == DType::I32 && rope_positions.data != nullptr &&
                rope_positions.is_contiguous() && rope_positions.ne[0] == tokens &&
                (rope_positions.ne[1] == 1 || rope_positions.ne[1] == 3),
            op, "rope positions must be I32 [W*B, 1|3]");
}

void require_index_geometry(const QsaGeometry& g, const char* op) {
    const bool rotary = g.rotary_dim == 32 || g.rotary_dim == 64 || g.rotary_dim == 128;
    bool axes         = true;
    for (int i = 0; i < g.rotary_dim / 2 && rotary; ++i) {
        axes = axes && g.pair_axes[static_cast<std::size_t>(i)] < 3;
    }
    require(g.index_dim == kQsaIndexKeyDim && g.index_heads >= 1 && g.index_heads <= 4 &&
                g.ratio == 4 && g.budget > 0 && g.budget % g.ratio == 0 && rotary && axes &&
                g.rope_theta > 0.0F && g.rms_norm_eps > 0.0F,
            op, "unsupported indexer geometry");
}

} // namespace

QsaGeometry qwen3_8_flash_next_qsa_geometry() {
    QsaGeometry g;
    g.query_heads  = 24;
    g.kv_heads     = 2;
    g.head_dim     = 256;
    g.index_heads  = 4;
    g.index_dim    = kQsaIndexKeyDim;
    g.budget       = 2048;
    g.ratio        = 4;
    g.rotary_dim   = 64;
    g.rope_theta   = 1.0e7F;
    g.rms_norm_eps = 1.0e-6F;
    // Interleaved MRoPE sections [11,11,10]: pair i rotates with axis i % 3.
    for (int i = 0; i < g.rotary_dim / 2; ++i) {
        g.pair_axes[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(i % 3);
    }
    return g;
}

void qsa_append(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                const Tensor& rope_positions, const Tensor& cache_positions,
                const Tensor& valid_columns, const Tensor& table_rows, const Tensor& key_norm,
                const QsaGeometry& geometry, const PagedKVBatchLayerView& kv,
                const QsaIndexPlanes& index, cudaStream_t stream) {
    constexpr const char* op = "qsa_append";
    require_storage(kv, op);
    require_index(index, op);
    require_index_geometry(geometry, op);
    require_dense(cache_positions, DType::I32, op, "cache positions");
    const std::int64_t tokens = cache_positions.numel();
    const std::int64_t lanes  = cache_positions.ne[1];
    require_dense(k, DType::BF16, op, "k");
    require_dense(v, DType::BF16, op, "v");
    require(k.ne[0] == 256 && k.ne[1] == kv.num_kv_heads && k.ne[2] == cache_positions.ne[0] &&
                k.ne[3] == lanes && v.numel() == k.numel(),
            op, "k/v must be [256,kv_heads,W,B]");
    require_dense(index_keys, DType::BF16, op, "index keys");
    require(index_keys.numel() == tokens * kQsaIndexKeyDim, op, "index keys must be [128,W,B]");
    require_positions(rope_positions, tokens, op);
    require_dense(table_rows, DType::I32, op, "table rows");
    require(table_rows.numel() == lanes, op, "one table row per lane");
    require_optional_lanes(valid_columns, lanes, op, "valid columns");
    require_dense(key_norm, DType::BF16, op, "key norm");
    require(key_norm.numel() == kQsaIndexKeyDim, op, "key norm must be BF16 [128]");
    detail::qsa_append_launch(k, v, index_keys, rope_positions, cache_positions, valid_columns,
                              table_rows, key_norm, geometry, kv, index, stream);
}

std::size_t qsa_select_workspace_bytes(const QsaGeometry& geometry, std::uint32_t max_visible_keys,
                                       std::int32_t width, std::int32_t lanes) {
    if (geometry.ratio <= 0 || width <= 0 || lanes <= 0 || max_visible_keys == 0) {
        throw std::invalid_argument("qsa_select_workspace_bytes: invalid geometry");
    }
    return detail::qsa_select_scratch_bytes(
               detail::qsa_select_plan(max_visible_keys, geometry.ratio, width, lanes)) +
           256;
}

void qsa_select(const Tensor& index_query, const Tensor& query_rope_positions,
                const Tensor& cache_positions, const Tensor& valid_columns,
                const Tensor& table_rows, const Tensor& query_norm, const QsaIndexPlanes& index,
                const QsaGeometry& geometry, std::uint32_t max_visible_keys,
                WorkspaceArena& workspace, Tensor& selected, Tensor& counts, cudaStream_t stream) {
    constexpr const char* op = "qsa_select";
    require_index_geometry(geometry, op);
    require_index(index, op);
    require(max_visible_keys > 0, op, "the visibility envelope must be positive");
    require_dense(cache_positions, DType::I32, op, "cache positions");
    const std::int32_t width   = cache_positions.ne[0];
    const std::int32_t lanes   = cache_positions.ne[1];
    const std::int64_t columns = cache_positions.numel();
    require(cache_positions.ne[2] == 1 && cache_positions.ne[3] == 1, op,
            "cache positions must be I32 [W,B]");
    require_dense(index_query, DType::BF16, op, "index query");
    require(index_query.ne[0] == geometry.index_heads * geometry.index_dim &&
                index_query.ne[1] == width && index_query.ne[2] == lanes,
            op, "index query must be [heads*128,W,B]");
    require_positions(query_rope_positions, columns, op);
    require_dense(table_rows, DType::I32, op, "table rows");
    require(table_rows.numel() == lanes, op, "one table row per lane");
    require_optional_lanes(valid_columns, lanes, op, "valid columns");
    require_dense(query_norm, DType::BF16, op, "query norm");
    require(query_norm.numel() == kQsaIndexKeyDim, op, "query norm must be BF16 [128]");
    require_dense(selected, DType::I32, op, "selected");
    require(selected.ne[0] == geometry.max_selected() && selected.ne[1] == width &&
                selected.ne[2] == lanes,
            op, "selected must be [max_selected,W,B]");
    require_dense(counts, DType::I32, op, "counts");
    require(counts.ne[0] == width && counts.ne[1] == lanes, op, "counts must be [W,B]");
    const detail::QsaSelectPlan plan =
        detail::qsa_select_plan(max_visible_keys, geometry.ratio, width, lanes);
    auto scope   = workspace.scope();
    auto scratch = workspace.alloc_bytes(detail::qsa_select_scratch_bytes(plan));
    detail::qsa_select_launch(index_query, query_rope_positions, cache_positions, valid_columns,
                              table_rows, query_norm, index, geometry, plan,
                              static_cast<float*>(scratch.data), selected, counts, stream);
}

std::size_t qsa_attention_workspace_bytes(const QsaGeometry& geometry, std::int32_t columns) {
    if (geometry.kv_heads <= 0 || geometry.query_heads % geometry.kv_heads != 0 || columns <= 0) {
        throw std::invalid_argument("qsa_attention_workspace_bytes: invalid geometry");
    }
    // Narrower calls split further, so size for every width up to `columns`.
    std::size_t bytes = 0;
    for (std::int32_t c = 1; c <= columns; ++c) {
        const std::size_t call = detail::qsa_attention_partial_bytes(
            c, geometry.query_heads, geometry.kv_heads, geometry.max_selected(), c);
        if (call == 0) { break; } // wider calls fill the GPU without splitting
        bytes = std::max(bytes, call);
    }
    return bytes + 256;
}

void qsa_attention(const Tensor& q, const Tensor& selected, const Tensor& counts,
                   const Tensor& table_rows, const PagedKVBatchLayerView& kv, float scale,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                   std::int32_t split_columns) {
    constexpr const char* op = "qsa_attention";
    require_storage(kv, op);
    require_dense(q, DType::BF16, op, "q");
    require(q.ne[0] == 256 && q.ne[1] % kv.num_kv_heads == 0 && q.ne[1] / kv.num_kv_heads <= 16, op,
            "q must be [256,heads,W,B] with at most 16 heads per KV head");
    require_dense(out, DType::BF16, op, "out");
    require(out.numel() == q.numel() && out.data != q.data, op, "out must match q and not alias");
    require_dense(selected, DType::I32, op, "selected");
    require(selected.ne[1] == q.ne[2] && selected.ne[2] == q.ne[3], op,
            "selected must be [max_selected,W,B]");
    require_dense(counts, DType::I32, op, "counts");
    require(counts.numel() == static_cast<std::int64_t>(q.ne[2]) * q.ne[3], op,
            "counts must be [W,B]");
    require_dense(table_rows, DType::I32, op, "table rows");
    require(table_rows.numel() == q.ne[3], op, "one table row per lane");
    require(std::isfinite(scale) && scale > 0.0F, op, "scale must be positive and finite");
    const std::int32_t columns = q.ne[2] * q.ne[3];
    require(split_columns == 0 || split_columns >= columns, op,
            "split_columns must be 0 or at least the call's columns");
    const std::int32_t policy = split_columns == 0 ? columns : split_columns;
    const std::size_t bytes   = detail::qsa_attention_partial_bytes(
        columns, q.ne[1], kv.num_kv_heads, static_cast<std::int32_t>(selected.ne[0]), policy);
    auto scope     = workspace.scope();
    float* partial = bytes == 0 ? nullptr : static_cast<float*>(workspace.alloc_bytes(bytes).data);
    detail::qsa_attention_launch(q, selected, counts, table_rows, kv, scale, partial, out, stream,
                                 policy);
}

} // namespace ninfer::ops
