#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/prepare/launch.h"
#include "ops/softmax_attention/qsa/validation.h"

#include "ops/softmax_attention/qsa/geometry.h"

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

void validate_kv(const detail::qsa::ViewValidation& v, const QsaKvProjectionView& p,
                 const QsaKvPrepareParameters& parameters, const QsaBatchView& batch,
                 const Tensor& rope_positions, const ConstQsaTailStateView& before,
                 QsaCacheView cache, QsaRawRecordWriteView records) {
    const int tokens = p.key.ne[2], rows = p.key.ne[3], axes = rope_positions.ne[1];
    const int kv_heads = p.key.ne[1];
    v.require(detail::qsa::registered_query_heads(kv_heads) != 0, "unsupported head geometry");
    v.require(tokens > 0 && rows >= 1 && rows <= 8 && (axes == 1 || axes == 3), "T/B/A");
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
    v.dense(rope_positions, DType::I32, {tokens, axes, rows, 1}, 4, "RoPE positions");
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
} // namespace

void qsa_prepare(const QsaProjectionView& p, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const Tensor& rope_positions,
                 const ConstQsaTailStateView& before, QsaCacheView cache,
                 QsaRawRecordWriteView records, Tensor& query, Tensor& index_query,
                 CudaExecutionView execution) {
    const detail::qsa::ViewValidation v{"qsa_prepare", execution.device()};
    validate_kv(v, {p.key, p.value, p.index_key}, parameters.kv, batch, rope_positions, before,
                cache, records);
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
                 &before.first_rope.records.byte_offsets, &records.offsets})
            v.require(!v.overlap(*out, *in), "output/parameter overlap");
    }
    detail::qsa::launch_prepare(p, parameters, batch, rope_positions, before, cache, records, query,
                                index_query, execution);
}

void qsa_prepare_kv(const QsaKvProjectionView& p, const QsaKvPrepareParameters& parameters,
                    const QsaBatchView& batch, const Tensor& rope_positions,
                    const ConstQsaTailStateView& before, QsaCacheView cache,
                    QsaRawRecordWriteView records, CudaExecutionView execution) {
    const detail::qsa::ViewValidation v{"qsa_prepare_kv", execution.device()};
    validate_kv(v, p, parameters, batch, rope_positions, before, cache, records);
    detail::qsa::launch_prepare_kv(p, parameters, batch, rope_positions, before, cache, records,
                                   execution);
}

} // namespace ninfer::ops
