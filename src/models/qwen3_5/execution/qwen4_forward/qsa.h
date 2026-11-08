#pragma once

// Qwen Sparse Attention (docs/maintainer/qwen4-exp-model.md#qwen-sparse-attention): the input
// projection (fused or one Linear and the row copies), the replicated indexer, the K/V and index
// append, the selection and the attention over it, also from active KV in RAM.
//
// Member definitions of Qwen4Forward (qwen4_forward.h), included only by qwen4_forward.cpp, which
// instantiates them for execution and for planning.

#include "models/qwen3_5/execution/qwen4_forward.h"
#include "models/qwen3_5/execution/attention.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/hyper_connection.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/sparse_attention.h"
#include "ninfer/ops/tp2/attn_input_proj.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <variant>

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] inline ops::QsaGeometry qwen4_qsa_geometry(const TextConfig& config) {
    const auto& attention = *config.attention;
    const auto& qsa       = *config.sparse_attention;
    const auto& rope      = *config.rope_parameters;
    ops::QsaGeometry g;
    g.query_heads  = dimension(attention.num_attention_heads);
    g.kv_heads     = dimension(attention.num_key_value_heads);
    g.head_dim     = dimension(attention.head_dim);
    g.index_heads  = dimension(qsa.indexer_heads);
    g.index_dim    = dimension(qsa.indexer_head_dim);
    g.budget       = dimension(qsa.budget);
    g.ratio        = dimension(qsa.compress_ratio);
    g.rotary_dim   = dimension(rope.rotary_dim);
    g.rope_theta   = rope.rope_theta;
    g.rms_norm_eps = config.rms_norm_eps;
    if (rope.pair_axes.size() > g.pair_axes.size()) {
        throw std::invalid_argument("QSA rotary pairs exceed the indexer geometry");
    }
    std::copy(rope.pair_axes.begin(), rope.pair_axes.end(), g.pair_axes.begin());
    return g;
}

template <class Arena>
void Qwen4Forward<Arena>::qsa_input_projection(std::size_t layer, const RankTensors& x,
                                               RankTensors& q, RankTensors& gate, RankTensors& k,
                                               RankTensors& v) {
    const auto p = [&](int r) -> const LinearParameters& {
        return std::get<QsaParameters>(block(r, layer).mixer).projection;
    };
    if constexpr (kExecute) {
        if (width_ == 1) {
            auto call = work(0).scope();
            ops::attn_input_proj(x[0], p(0).weight, q[0], gate[0], k[0], v[0], p(0).policy, work(0),
                                 s(0));
        } else {
            const std::array<WorkspaceArena*, 2> ws{&work(0), &work(1)};
            ops::attn_input_proj_column_parallel(x, {p(0).weight, p(1).weight}, q, gate, k, v,
                                                 p(0).policy, ws, *tp_.execution);
        }
    } else {
        (void)q, (void)gate, (void)k, (void)v, (void)x;
        for (int r = 0; r < width_; ++r) {
            const LinearParameters& w = p(r);
            reserve(width_ == 1
                        ? ops::attn_input_proj_workspace_capacity_bytes(
                              w.weight.qtype, w.weight.n, w.weight.k, w.policy, first_, last_)
                        : ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                              w.weight.qtype, w.policy, first_, last_),
                    r);
        }
    }
}

template <class Arena>
bool Qwen4Forward<Arena>::kv_in_host() const {
    if constexpr (kExecute) {
        return state(0).kv != nullptr && state(0).kv->qsa_staging() != nullptr;
    } else {
        return c(0).kv_ram_active;
    }
}

template <class Arena>
bool Qwen4Forward<Arena>::staging_stats() {
    static const bool on = [] {
        const char* env = std::getenv("NINFER_KVRAM_STATS");
        return env != nullptr && std::string_view(env) == "1";
    }();
    return on;
}

// The operands of one QSA call on every rank, allocated in this order by qsa_buffers (the
// planned workspace replays the same order).
template <class Arena>
struct Qwen4Forward<Arena>::QsaBuffers {
    // Fused route (Qwen4FusedRoutes::attn_input): q, k, gate, v written directly by the 27B's
    // input-projection Op; otherwise one Linear into `projection` and the row copies.
    bool fused = false;
    ops::QsaGeometry geometry;
    RankTensors projection, q, k, gate, v, qn, kn, indexer, index_query, index_key;
    RankTensors selected, counts, attended;
    // Active KV in RAM: a decode window reads the selection rewritten onto the VRAM region.
    RankTensors rewritten;
};

template <class Arena>
void Qwen4Forward<Arena>::qsa(std::size_t layer, const RankTensors& x, RankTensors* y,
                              std::uint32_t kv_layer, const RankTensors& staging) {
    const std::int32_t T = columns();
    const auto qw        = dimension(rc().attention->query_width());
    const auto p         = [&](int r) -> const QsaParameters& {
        return std::get<QsaParameters>(block(r, layer).mixer);
    };
    auto scope   = scopes();
    QsaBuffers b = qsa_buffers(layer, y != nullptr);
    each([&](int r) {
        if (!b.fused) { linear(r, x[at(r)], p(r).projection, b.projection[at(r)]); }
        linear(r, x[at(r)], p(r).indexer, b.indexer[at(r)]);
    });
    if (b.fused) { qsa_input_projection(layer, x, b.q, b.gate, b.k, b.v); }
    if constexpr (!kExecute) {
        // Selection scratch for the planned window: W columns of B lanes.
        const std::int32_t W = c(0).width, B = c(0).batch;
        reserve(ops::qsa_select_workspace_bytes(b.geometry, std::max(c(0).max_visible, 1U), W, B));
        reserve(ops::qsa_attention_workspace_bytes(b.geometry, last_));
        if (kv_in_host() && c(0).prefill) {
            const std::uint32_t visible = std::max(c(0).max_visible, 1U);
            reserve(std::max(ops::qsa_count_selected_blocks_workspace_bytes(visible),
                             ops::qsa_stage_selected_union_workspace_bytes(visible)));
        }
    } else {
        each([&](int r) { qsa_rank(layer, r, b, kv_layer, y != nullptr); });
    }
    if (y == nullptr) { return; }
    const RankTensors flat{b.attended[0].view({qw, T}),
                           width_ == 2 ? b.attended[1].view({qw, T}) : Tensor{}};
    output_projection(flat, {&p(0).output, width_ == 2 ? &p(1).output : nullptr}, *y, staging,
                      static_cast<std::int32_t>(layer));
}

template <class Arena>
auto Qwen4Forward<Arena>::qsa_buffers(std::size_t layer, bool attend) -> QsaBuffers {
    const std::int32_t T = columns();
    const auto& a        = *rc().attention;
    const auto head_dim  = dimension(a.head_dim);
    const auto heads     = dimension(a.num_attention_heads);
    const auto kv_heads  = dimension(a.num_key_value_heads);
    const auto qw        = dimension(a.query_width());
    const auto kw        = dimension(a.key_width());
    QsaBuffers b;
    b.fused               = block(0, layer).fused.attn_input;
    b.geometry            = qwen4_qsa_geometry(rc());
    const auto& g         = b.geometry;
    const auto index_rows = g.index_heads * g.index_dim;
    if (!b.fused) { b.projection = alloc(DType::BF16, {2 * qw + 2 * kw, T}); }
    b.q                  = alloc(DType::BF16, {qw, T});
    b.k                  = alloc(DType::BF16, {kw, T});
    b.gate               = alloc(DType::BF16, {qw, T});
    b.v                  = alloc(DType::BF16, {kw, T});
    b.qn                 = alloc(DType::BF16, {head_dim, heads, T});
    b.kn                 = alloc(DType::BF16, {head_dim, kv_heads, T});
    b.indexer            = alloc(DType::BF16, {index_rows + g.index_dim, T});
    b.index_query        = alloc(DType::BF16, {index_rows, T});
    b.index_key          = alloc(DType::BF16, {g.index_dim, T});
    const std::int32_t W = c(0).width, B = c(0).batch;
    b.selected = alloc(DType::I32, {g.max_selected(), kExecute ? W : T, kExecute ? B : 1});
    b.counts   = alloc(DType::I32, {kExecute ? W : T, kExecute ? B : 1});
    b.attended = alloc(DType::BF16, {head_dim, heads, T});
    // Active KV in RAM: decode windows, and prompt chunks beyond the region (union staging), read
    // the selection rewritten onto the region.
    if (kv_in_host() && attend) {
        b.rewritten = alloc(DType::I32, {g.max_selected(), kExecute ? W : T, kExecute ? B : 1});
    }
    return b;
}

template <class Arena>
void Qwen4Forward<Arena>::qsa_rank(std::size_t layer, int r, QsaBuffers& b, std::uint32_t kv_layer,
                                   bool attend) {
    if constexpr (kExecute) {
        const std::int32_t T  = columns();
        const auto& a         = *rc().attention;
        const auto head_dim   = dimension(a.head_dim);
        const auto heads      = dimension(a.num_attention_heads);
        const auto kv_heads   = dimension(a.num_key_value_heads);
        const auto qw         = dimension(a.query_width());
        const auto kw         = dimension(a.key_width());
        const auto& geometry  = b.geometry;
        const auto index_rows = geometry.index_heads * geometry.index_dim;
        const bool fused      = b.fused;
        const std::int32_t W = c(0).width, B = c(0).batch;
        const QsaParameters& qp = std::get<QsaParameters>(block(r, layer).mixer);
        const std::size_t i     = at(r);
        const Qwen4Columns& col = c(r);
        cudaStream_t stream     = s(r);
        const Tensor qs         = fused ? b.q[i] : split_rows(r, b.projection[i], 0, qw, b.q[i]);
        const Tensor ks         = fused ? b.k[i] : split_rows(r, b.projection[i], qw, kw, b.k[i]);
        const Tensor gs =
            fused ? b.gate[i] : split_rows(r, b.projection[i], qw + kw, qw, b.gate[i]);
        const Tensor vs = fused ? b.v[i] : split_rows(r, b.projection[i], 2 * qw + kw, kw, b.v[i]);
        const Tensor iq = split_rows(r, b.indexer[i], 0, index_rows, b.index_query[i]);
        const Tensor ik =
            split_rows(r, b.indexer[i], index_rows, geometry.index_dim, b.index_key[i]);
        if (attend) {
            ops::rmsnorm(qs.view({head_dim, heads, T}), qp.query_norm, config_.rms_norm_eps, true,
                         b.qn[i], stream);
        }
        ops::rmsnorm(ks.view({head_dim, kv_heads, T}), qp.key_norm, config_.rms_norm_eps, true,
                     b.kn[i], stream);
        const Tensor& rope = *col.rope_positions;
        // Only prefill binds axis-major [T,3] MRoPE rows; decode windows bind [W,B]
        // positions.
        const bool mrope       = col.prefill && rope.ne[1] == 3;
        const Tensor rope_flat = mrope ? rope : rope.view({T});
        if (attend) {
            text_rope(rope_flat, *config_.rope_parameters, b.qn[i], b.kn[i], stream);
        } else {
            text_rope(rope_flat, *config_.rope_parameters, b.kn[i], stream);
        }
        const Tensor rope_rows = mrope ? rope : rope.view({T, 1});
        const Tensor positions = col.cache_positions->view({W, B});
        const Tensor valid     = col.valid_columns != nullptr ? *col.valid_columns : Tensor{};
        const PagedKVBatchLayerView kv  = state(r).kv->batch_layer_view(kv_layer);
        const ops::QsaIndexPlanes index = state(r).kv->qsa_index_planes(kv_layer);
        ops::qsa_append(b.kn[i].view({head_dim, kv_heads, W, B}),
                        vs.view({head_dim, kv_heads, W, B}), ik.view({geometry.index_dim, W, B}),
                        rope_rows, positions, valid, *col.table_rows, qp.indexer_key_norm, geometry,
                        kv, index, stream);
        if (!attend) { return; }
        ops::qsa_select(iq.view({index_rows, W, B}), rope_rows, positions, valid, *col.table_rows,
                        qp.indexer_query_norm, index, geometry, col.max_visible, work(r),
                        b.selected[i], b.counts[i], stream);
        qsa_attend(r, b, kv, positions, valid, kv_layer);
        Tensor flat = b.attended[i].view({qw, T});
        ops::sigmoid_mul(gs, flat, stream);
    } else {
        (void)layer, (void)r, (void)b, (void)kv_layer, (void)attend;
    }
}

template <class Arena>
void Qwen4Forward<Arena>::qsa_attend(int r, QsaBuffers& b, const PagedKVBatchLayerView& kv,
                                     const Tensor& positions, const Tensor& valid,
                                     std::uint32_t kv_layer) {
    if constexpr (kExecute) {
        const auto& a        = *rc().attention;
        const auto head_dim  = dimension(a.head_dim);
        const auto heads     = dimension(a.num_attention_heads);
        const auto& geometry = b.geometry;
        const std::int32_t W = c(0).width, B = c(0).batch;
        const std::size_t i     = at(r);
        const Qwen4Columns& col = c(r);
        cudaStream_t stream     = s(r);
        Tensor a4               = b.attended[i].view({head_dim, heads, W, B});
        const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim)));
        if (const ops::QsaStagingRegion* region = state(r).kv->qsa_staging()) {
            // Active KV in RAM: the selected keys come from the Host planes into the
            // region (bit-identical bytes); the attention reads the region.
            if (geometry.ratio != 4 || (col.prefill && B != 1)) {
                throw std::logic_error("active KV in RAM serves 4-token QSA blocks and prefills "
                                       "one lane at a time");
            }
            // Lane b reads its rewritten selection through identity row entry b.
            const Tensor identity = region->table_row.slice(0, 0, B);
            if (col.prefill) {
                if (staging_stats() && state(r).kv->staging_layer(kv_layer) == 0) {
                    state(r).kv->report_staging_stats("before prefill");
                }
                if (staging_stats()) {
                    ops::qsa_count_selected_blocks(
                        b.selected[i], b.counts[i], positions, col.max_visible, *region,
                        state(r).kv->staging_layer(kv_layer), work(r), stream);
                }
                // Whole visible pages while the region holds them and the chunk may select as
                // many blocks as there are (long chunks select 73-95 % of them); otherwise the
                // union of the selections (a resumed conversation's short suffix touches a few
                // per cent of a long context).
                const std::int32_t groups = (geometry.max_selected() + 3) / 4;
                const std::int64_t visible_pages =
                    (static_cast<std::int64_t>(col.max_visible) + kPagedKVPageSize - 1) /
                    kPagedKVPageSize;
                const std::int64_t visible_blocks =
                    (static_cast<std::int64_t>(col.max_visible) + 3) / 4;
                if (visible_pages <= region->total_pages() &&
                    static_cast<std::int64_t>(W) * groups >= visible_blocks) {
                    ops::qsa_stage_visible_pages(kv, positions, valid, *col.table_rows,
                                                 col.max_visible, *region, staging_stats(), stream);
                    ops::qsa_attention(b.qn[i].view({head_dim, heads, W, B}), b.selected[i],
                                       b.counts[i], identity, region->pages, scale, work(r), a4,
                                       stream);
                } else {
                    // The union of the selections of as many columns as the region surely
                    // holds, then their attention, column range by column range.
                    const auto step = static_cast<std::int32_t>(std::max<std::int64_t>(
                        1, ops::qsa_staging_region_blocks(*region) / groups));
                    const Tensor q4 = b.qn[i].view({head_dim, heads, W, B});
                    for (std::int32_t w0 = 0; w0 < W; w0 += step) {
                        const std::int32_t n = std::min(step, W - w0);
                        Tensor rewritten     = b.rewritten[i].slice(1, w0, n);
                        Tensor out           = a4.slice(2, w0, n);
                        const Tensor counts  = b.counts[i].slice(0, w0, n);
                        ops::qsa_stage_selected_union(b.selected[i].slice(1, w0, n), counts,
                                                      *col.table_rows, kv, col.max_visible, *region,
                                                      staging_stats(), work(r), rewritten, stream);
                        // Split the keys as the whole chunk would: bit-identical columns.
                        ops::qsa_attention(q4.slice(2, w0, n), rewritten, counts, identity,
                                           region->pages, scale, work(r), out, stream, W * B);
                    }
                }
            } else {
                ops::qsa_stage_selected_blocks(
                    b.selected[i], b.counts[i], positions, *col.table_rows, kv, *region,
                    state(r).kv->staging_layer(kv_layer), staging_stats(), b.rewritten[i], stream);
                ops::qsa_attention(b.qn[i].view({head_dim, heads, W, B}), b.rewritten[i],
                                   b.counts[i], identity, region->pages, scale, work(r), a4,
                                   stream);
            }
        } else {
            ops::qsa_attention(b.qn[i].view({head_dim, heads, W, B}), b.selected[i], b.counts[i],
                               *col.table_rows, kv, scale, work(r), a4, stream);
        }
    } else {
        (void)r, (void)b, (void)kv, (void)positions, (void)valid, (void)kv_layer;
    }
}

} // namespace ninfer::models::qwen3_5::execution
