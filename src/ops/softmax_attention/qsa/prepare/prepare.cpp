#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/prepare/launch.h"
#include "ops/softmax_attention/qsa/validation.h"

#include "ops/softmax_attention/qsa/geometry.h"

#include "core/layout.h"

#include <cmath>
#include <initializer_list>

namespace ninfer::ops {
namespace {

bool identical(const Tensor& a, const Tensor& b) {
    if (a.data != b.data || a.device != b.device || a.dtype != b.dtype) return false;
    for (int d = 0; d < 4; ++d)
        if (a.ne[d] != b.ne[d] || a.nb[d] != b.nb[d]) return false;
    return true;
}

template <bool StridedRope>
void validate_kv(const detail::qsa::ViewValidation& v, const QsaKvProjectionView& p,
                 const QsaKvPrepareParameters& parameters, const QsaBatchView& batch,
                 const Tensor& rope_positions, const ConstQsaTailStateView& before,
                 QsaCacheView cache, QsaRawRecordWriteView records) {
    const int tokens = p.key.ne[2], rows = p.key.ne[3], axes = rope_positions.ne[1];
    const int kv_heads = p.key.ne[1];
    v.require(detail::qsa::registered_query_heads(kv_heads) != 0, "unsupported head geometry");
    v.require(tokens > 0 && rows >= 1 && (axes == 1 || axes == 3), "T/B/A");
    v.require(parameters.rms_epsilon > 0 && std::isfinite(parameters.rms_epsilon) &&
                  parameters.rope_theta > 0 && std::isfinite(parameters.rope_theta),
              "epsilon/theta");
    v.require(parameters.rotary_dim == 64 && parameters.rope_interleaved &&
                  parameters.rope_sections == std::array<std::uint32_t, 3>{11, 11, 10},
              "unsupported rotary geometry");
    v.dense(p.key, DType::BF16, {256, kv_heads, tokens, rows}, 16, "key projection");
    v.dense(p.value, DType::BF16, {256, kv_heads, tokens, rows}, 16, "value projection");
    v.dense(p.index_key, DType::BF16, {128, tokens, rows, 1}, 16, "index-key projection");
    v.dense(parameters.key_norm, DType::BF16, {256, 1, 1, 1}, 2, "key norm");
    v.dense(parameters.index_key_norm, DType::BF16, {128, 1, 1, 1}, 2, "index-key norm");
    v.dense(batch.positions, DType::I32, {tokens, rows, 1, 1}, 4, "positions");
    if constexpr (StridedRope) {
        v.require(rows == 1, "strided coordinate rows");
        v.columns(rope_positions, DType::I32, tokens, axes, rows, "RoPE positions");
    } else {
        v.dense(rope_positions, DType::I32, {tokens, axes, rows, 1}, 4, "RoPE positions");
    }
    v.metadata(batch.valid_columns, rows, "valid columns");
    v.metadata(batch.table_rows, rows, "table rows");
    v.metadata(records.offsets, rows, "record offsets");
    v.state(before, rows);
    const int capacity = records.raw_keys.ne[1];
    v.columns(records.raw_keys, DType::BF16, 128, capacity, rows, "raw records");
    v.columns(records.rope_positions, DType::I32, 3, capacity, rows, "coordinate records");

    const auto& main = cache.main;
    v.require(main.head_dim == 256 && main.num_kv_heads == kv_heads, "cache geometry");
    const auto layout        = paged_kv_storage_layout(main.storage, 256);
    const auto vector_planes = [&](const PagedPlaneView& data, const PagedPlaneView& scale,
                                   const PagedKVVectorLayout& schema) {
        common::operand::paged_plane(data, schema.data_dtype,
                                     {schema.data_leading_extent, 64, kv_heads},
                                     dtype_size(schema.data_dtype), v.op, "cache data", v.device);
        common::operand::same_paged_records(main.k_pages, data, v.op, "cache");
        v.require(data.records.page_tokens == 64 && data.plane.geometry.tokens_per_row == 1,
                  "main record geometry");
        if (schema.has_scale()) {
            common::operand::paged_plane(
                scale, schema.scale_dtype, {schema.scale_leading_extent, 64, kv_heads},
                dtype_size(schema.scale_dtype), v.op, "cache scale", v.device);
            common::operand::same_paged_records(data, scale, v.op, "scales");
        } else {
            v.require(scale.data() == nullptr, "absent cache scale");
        }
    };
    vector_planes(main.k_pages, main.k_scale_pages, layout.key);
    vector_planes(main.v_pages, main.v_scale_pages, layout.value);
    common::operand::paged_plane(cache.index.keys, DType::BF16, {128, 16, 1}, 2, v.op,
                                 "index pages", v.device);
    common::operand::same_paged_records(main.k_pages, cache.index.keys, v.op, "index cache");
    v.require(cache.index.keys.plane.geometry.tokens_per_row == 4, "index record geometry");
}

template <bool StridedRope>
void prepare(const QsaProjectionView& p, const QsaPrepareParameters& parameters,
             const QsaBatchView& batch, const Tensor& rope_positions,
             const ConstQsaTailStateView& before, QsaCacheView cache, QsaRawRecordWriteView records,
             Tensor& query, Tensor& index_query, CudaExecutionView execution) {
    const detail::qsa::ViewValidation v{"qsa_prepare", execution.device()};
    validate_kv<StridedRope>(v, {p.key, p.value, p.index_key}, parameters.kv, batch, rope_positions,
                             before, cache, records);
    const int tokens = p.key.ne[2], rows = p.key.ne[3],
              query_heads = detail::qsa::registered_query_heads(p.key.ne[1]);
    v.dense(p.query, DType::BF16, {256, query_heads, tokens, rows}, 16, "query projection");
    v.dense(p.index_query, DType::BF16, {128, 4, tokens, rows}, 16, "index-query projection");
    v.dense(query, DType::BF16, {256, query_heads, tokens, rows}, 16, "query output");
    v.dense(index_query, DType::BF16, {128, 4, tokens, rows}, 16, "index-query output");
    v.dense(parameters.query_norm, DType::BF16, {256, 1, 1, 1}, 2, "query norm");
    v.dense(parameters.index_query_norm, DType::BF16, {128, 1, 1, 1}, 2, "index-query norm");
    v.require(!v.overlap(query, index_query), "output overlap");
    for (const Tensor* out : {&query, &index_query}) {
        for (const Tensor* in : {&p.query, &p.key, &p.value, &p.index_query, &p.index_key}) {
            const bool allowed =
                (out == &query && in == &p.query) || (out == &index_query && in == &p.index_query);
            v.require((allowed && identical(*out, *in)) || !v.overlap(*out, *in),
                      "projection alias");
        }
        for (const Tensor* in : std::initializer_list<const Tensor*>{
                 &parameters.query_norm, &parameters.kv.key_norm, &parameters.index_query_norm,
                 &parameters.kv.index_key_norm, &batch.positions, &batch.valid_columns,
                 &batch.table_rows, &rope_positions, &before.raw_keys.records.byte_offsets,
                 &before.first_rope.records.byte_offsets, &records.offsets}) {
            if constexpr (StridedRope) {
                if (in == &rope_positions) {
                    v.require(
                        !common::operand::overlaps(common::operand::blocks(rope_positions, v.op),
                                                   common::operand::range(*out, v.op)),
                        "output/coordinate overlap");
                    continue;
                }
            }
            v.require(!v.overlap(*out, *in), "output/parameter overlap");
        }
    }
    detail::qsa::launch_prepare<StridedRope>(p, parameters, batch, rope_positions, before, cache,
                                             records, query, index_query, execution);
}

template <bool StridedRope>
void prepare_kv(const QsaKvProjectionView& p, const QsaKvPrepareParameters& parameters,
                const QsaBatchView& batch, const Tensor& rope_positions,
                const ConstQsaTailStateView& before, QsaCacheView cache,
                QsaRawRecordWriteView records, CudaExecutionView execution) {
    const detail::qsa::ViewValidation v{"qsa_prepare_kv", execution.device()};
    validate_kv<StridedRope>(v, p, parameters, batch, rope_positions, before, cache, records);
    detail::qsa::launch_prepare_kv<StridedRope>(p, parameters, batch, rope_positions, before, cache,
                                                records, execution);
}


} // namespace

void qsa_prepare(const QsaProjectionView& p, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const Tensor& rope_positions,
                 const ConstQsaTailStateView& before, QsaCacheView cache,
                 QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                 CudaExecutionView execution) {
    prepare<false>(p, parameters, batch, rope_positions, before, cache, records, query, index_query,
                   execution);
}

void qsa_prepare_kv(const QsaKvProjectionView& p, const QsaKvPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const ConstQsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, CudaExecutionView execution) {
    prepare_kv<false>(p, parameters, batch, rope_positions, before, cache, records, execution);
}

namespace {
namespace operand                       = common::operand;
constexpr const char* ragged_prepare_op = "qsa_prepare_update";

template <class Arena>
QsaRawRecordWriteView allocate_prepare_records(Arena& arena, int width, Tensor offsets = {}) {
    return {arena.alloc(DType::BF16, {128, width, 1}, 256),
            arena.alloc(DType::I32, {3, width, 1}, 256), offsets};
}

QsaKvProjectionView kv_columns(const QsaKvProjectionView& p, int begin, int length) {
    return {p.key.slice(2, begin, length), p.value.slice(2, begin, length),
            p.index_key.slice(1, begin, length)};
}

template <class Prepare>
void prepare_update(const QsaKvProjectionView& kv, const QsaKvPrepareParameters& params,
                    const QsaRaggedBatchView& batch, const Tensor& rope,
                    const ConstQsaTailStateView& before, QsaTailStateView after,
                    MemoryArena& workspace, std::span<const Tensor* const> queries, Prepare prepare,
                    CudaExecutionView execution) {
    const auto& seq         = batch.ragged;
    const auto device       = execution.device();
    constexpr const auto op = ragged_prepare_op;
    operand::ragged_batch(seq, batch.positions, batch.table_rows, op, device);
    operand::device(workspace.device(), device, op, "workspace");
    const int n = seq.total_tokens(), axes = rope.ne[1], heads = kv.key.ne[1];
    const detail::qsa::ViewValidation v{op, device};
    v.require(axes == 1 || axes == 3, "coordinate axes");
    v.columns(rope, DType::I32, n, axes, 1, "RoPE positions");
    v.state(before, seq.num_seqs());
    v.state(after, seq.num_seqs());
    operand::tensor(kv.key, DType::BF16, {256, heads, n}, 16, op, "key projection", device);
    operand::tensor(kv.value, DType::BF16, {256, heads, n}, 16, op, "value projection", device);
    operand::tensor(kv.index_key, DType::BF16, {128, n}, 16, op, "index-key projection", device);
    const Tensor* tensors[] = {&kv.key,
                               &kv.value,
                               &kv.index_key,
                               &params.key_norm,
                               &params.index_key_norm,
                               &batch.positions,
                               &batch.table_rows,
                               &seq.cu_seqlens,
                               &seq.seq_lengths,
                               &seq.token_to_seq,
                               &before.raw_keys.records.byte_offsets,
                               &before.first_rope.records.byte_offsets,
                               &after.raw_keys.records.byte_offsets,
                               &after.first_rope.records.byte_offsets};
    const auto coordinates  = operand::blocks(rope, op);
    for (int s = 0; s < seq.num_seqs(); ++s) {
        const int begin = seq.segment_begin(s), len = seq.segment_length(s);
        auto scope = workspace.scope();
        const QsaBatchView segment{batch.positions.slice(0, begin, len),
                                   seq.seq_lengths.slice(0, s, 1), batch.table_rows.slice(0, s, 1)};
        const ConstQsaTailStateView src{operand::state_row(before.raw_keys, s),
                                        operand::state_row(before.first_rope, s)};
        const QsaTailStateView dst{operand::state_row(after.raw_keys, s),
                                   operand::state_row(after.first_rope, s)};
        const auto records =
            allocate_prepare_records(workspace, len, seq.cu_seqlens.slice(0, 0, 1));
        for (const auto* t : {&records.raw_keys, &records.rope_positions}) {
            const auto scratch = operand::range(*t, op);
            v.require(!operand::overlaps(coordinates, scratch), "workspace/coordinate overlap");
            for (const auto* input : tensors)
                operand::disjoint(scratch, operand::range(*input, op), op);
            for (const auto* input : queries)
                operand::disjoint(scratch, operand::range(*input, op), op);
        }
        prepare(kv_columns(kv, begin, len), segment, rope.slice(0, begin, len), src, records, begin,
                len);
        qsa_tail_extract(src,
                         {records.raw_keys, records.rope_positions,
                          batch.positions.slice(0, begin, 1), segment.valid_columns},
                         segment.valid_columns, dst, {}, execution);
    }
}
} // namespace

std::size_t qsa_prepare_kv_update_ragged_workspace_capacity_bytes(int min_tokens, int max_tokens,
                                                                  int min_segments,
                                                                  int max_segments) {
    const auto widths = operand::ragged_envelope(min_tokens, max_tokens, min_segments, max_segments,
                                                 ragged_prepare_op);
    WorkspaceLayoutBuilder layout;
    (void)allocate_prepare_records(layout, widths.last);
    return layout.peak_bytes(1);
}

std::size_t qsa_prepare_update_ragged_workspace_capacity_bytes(int min_tokens, int max_tokens,
                                                               int min_segments, int max_segments) {
    return qsa_prepare_kv_update_ragged_workspace_capacity_bytes(min_tokens, max_tokens,
                                                                 min_segments, max_segments);
}

void qsa_prepare_update(const QsaProjectionView& p, const QsaPrepareParameters& parameters,
                        const QsaRaggedBatchView& batch, const Tensor& rope,
                        const ConstQsaTailStateView& before, QsaCacheView cache,
                        QsaTailStateView after, Tensor& query, Tensor& index_query,
                        MemoryArena& workspace, CudaExecutionView execution) {
    const int n = batch.ragged.total_tokens(), h = detail::qsa::registered_query_heads(p.key.ne[1]);
    const auto device       = execution.device();
    constexpr const auto op = ragged_prepare_op;
    for (const auto* t : std::array<const Tensor*, 2>{&p.query, &query})
        operand::tensor(*t, DType::BF16, {256, h, n}, 16, op, "query", device);
    for (const auto* t : std::array<const Tensor*, 2>{&p.index_query, &index_query})
        operand::tensor(*t, DType::BF16, {128, 4, n}, 16, op, "index query", device);
    const Tensor* queries[] = {&p.query,     &p.index_query,         &query,
                               &index_query, &parameters.query_norm, &parameters.index_query_norm};
    prepare_update(
        {p.key, p.value, p.index_key}, parameters.kv, batch, rope, before, after, workspace,
        queries,
        [&](const QsaKvProjectionView& kv, const QsaBatchView& segment, const Tensor& coords,
            const ConstQsaTailStateView& src, QsaRawRecordWriteView records, int begin, int len) {
            auto out = query.slice(2, begin, len);
            auto iq  = index_query.slice(2, begin, len);
            prepare<true>({p.query.slice(2, begin, len), kv.key, kv.value,
                           p.index_query.slice(2, begin, len), kv.index_key},
                          parameters, segment, coords, src, cache, records, out, iq, execution);
        },
        execution);
}

void qsa_prepare_kv_update(const QsaKvProjectionView& p, const QsaKvPrepareParameters& parameters,
                           const QsaRaggedBatchView& batch, const Tensor& rope,
                           const ConstQsaTailStateView& before, QsaCacheView cache,
                           QsaTailStateView after, MemoryArena& workspace,
                           CudaExecutionView execution) {
    prepare_update(
        p, parameters, batch, rope, before, after, workspace, {},
        [&](const QsaKvProjectionView& kv, const QsaBatchView& segment, const Tensor& coords,
            const ConstQsaTailStateView& src, QsaRawRecordWriteView records, int, int) {
            prepare_kv<true>(kv, parameters, segment, coords, src, cache, records, execution);
        },
        execution);
}

} // namespace ninfer::ops
