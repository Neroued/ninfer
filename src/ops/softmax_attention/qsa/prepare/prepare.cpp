#include "ninfer/ops/qsa.h"
#include "ops/softmax_attention/qsa/prepare/launch.h"
#include "ops/softmax_attention/qsa/validation.h"

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

} // namespace

void qsa_prepare(const QsaProjectionView& p, const QsaPrepareParameters& parameters,
                 const QsaBatchView& batch, const Tensor& rope_positions,
                 const QsaTailStateView& before, QsaCacheView cache, QsaRawRecordWriteView records,
                 Tensor& query, Tensor& index_query, CudaExecutionView execution) {
    const detail::qsa::ViewValidation v{"qsa_prepare", execution.device()};
    const int tokens = p.query.ne[2], rows = p.query.ne[3], axes = rope_positions.ne[1];
    const int kv_heads = p.key.ne[1];
    v.require(kv_heads == 1 || kv_heads == 2, "unsupported head geometry");
    const int query_heads = 12 * kv_heads;
    v.require(tokens > 0 && rows >= 1 && rows <= 8 && (axes == 1 || axes == 3), "T/B/A");
    v.require(parameters.rms_epsilon > 0 && std::isfinite(parameters.rms_epsilon) &&
                  parameters.rope_theta > 0 && std::isfinite(parameters.rope_theta),
              "epsilon/theta");
    v.require(parameters.rotary_dim == 64 && parameters.rope_interleaved &&
                  parameters.rope_sections == std::array<std::uint32_t, 3>{11, 11, 10},
              "unsupported rotary geometry");
    v.dense(p.query, DType::BF16, {256, query_heads, tokens, rows}, 16, "query projection");
    v.dense(p.key, DType::BF16, {256, kv_heads, tokens, rows}, 16, "key projection");
    v.dense(p.value, DType::BF16, {256, kv_heads, tokens, rows}, 16, "value projection");
    v.dense(p.index_query, DType::BF16, {128, 4, tokens, rows}, 16, "index-query projection");
    v.dense(p.index_key, DType::BF16, {128, tokens, rows, 1}, 16, "index-key projection");
    v.dense(query, DType::BF16, {256, query_heads, tokens, rows}, 16, "query output");
    v.dense(index_query, DType::BF16, {128, 4, tokens, rows}, 16, "index-query output");
    v.dense(parameters.query_norm, DType::BF16, {256, 1, 1, 1}, 2, "query norm");
    v.dense(parameters.key_norm, DType::BF16, {256, 1, 1, 1}, 2, "key norm");
    v.dense(parameters.index_query_norm, DType::BF16, {128, 1, 1, 1}, 2, "index-query norm");
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
    const int pages  = main.k_pages.ne[3];
    v.require(main.head_dim == 256 && main.num_kv_heads == kv_heads, "cache geometry");
    const auto layout        = paged_kv_storage_layout(main.storage, 256);
    const auto vector_planes = [&](const Tensor& data, const Tensor& scale,
                                   const PagedKVVectorLayout& schema) {
        v.plane(data, schema.data_dtype, {schema.data_leading_extent, 64, kv_heads, pages},
                "cache data");
        if (schema.has_scale())
            v.plane(scale, schema.scale_dtype, {schema.scale_leading_extent, 64, kv_heads, pages},
                    "cache scale");
        else
            v.require(scale.data == nullptr, "absent cache scale");
    };
    vector_planes(main.k_pages, main.k_scale_pages, layout.key);
    vector_planes(main.v_pages, main.v_scale_pages, layout.value);
    v.plane(cache.index.keys, DType::BF16, {128, 16, 1, pages}, "index pages");
    v.plane(main.block_tables, DType::I32, {main.block_tables.ne[0], main.block_tables.ne[1], 1, 1},
            "block tables");
    v.require(identical(main.block_tables, cache.index.block_tables), "shared block tables");

    v.require(!v.overlap(query, index_query), "output overlap");
    for (const Tensor* out : {&query, &index_query}) {
        for (const Tensor* in : {&p.query, &p.key, &p.value, &p.index_query, &p.index_key}) {
            const bool allowed =
                (out == &query && in == &p.query) || (out == &index_query && in == &p.index_query);
            v.require((allowed && identical(*out, *in)) || !v.overlap(*out, *in),
                      "projection alias");
        }
        for (const Tensor* in : std::initializer_list<const Tensor*>{
                 &parameters.query_norm, &parameters.key_norm, &parameters.index_query_norm,
                 &parameters.index_key_norm, &batch.positions, &batch.valid_columns,
                 &batch.table_rows, &rope_positions, &before.slots, &records.offsets})
            v.require(!v.overlap(*out, *in), "output/parameter overlap");
    }
    detail::qsa::launch_prepare(p, parameters, batch, rope_positions, before, cache, records, query,
                                index_query, execution);
}

} // namespace ninfer::ops
