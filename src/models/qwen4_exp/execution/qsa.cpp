#include "models/qwen4_exp/execution/qsa.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "ninfer/ops/linear_split.h"

namespace ninfer::models::qwen4_exp::execution {
namespace {
void prepare(const Tensor& input, const QsaConfig& c, const QsaParameters& p,
             const BatchView& batch, const ops::QsaTailStateView& before, ops::QsaCacheView cache,
             workspace::QsaBuffers& buffers, ops::QsaRawRecordWriteView& records,
             MemoryArena& arena, CudaExecutionView cuda) {
    const auto d = dimension(c.head_dim), qh = dimension(c.query_heads), kh = dimension(c.kv_heads);
    const auto id = dimension(c.index_dim), ih = dimension(c.index_heads);
    const auto qw = dimension(c.query_width()), kw = dimension(c.kv_width());
    const auto iq = dimension(c.index_query_width()), ik = dimension(c.index_key_width());
    const auto t = batch.tokens, b = batch.batch, m = t * b;
    if (!records.raw_keys.data)
        records = {buffers.raw_records.view({ik, t, b}), buffers.coordinate_records.view({3, t, b}),
                   batch.record_offsets};
    {
        auto scratch = arena.scope();
        if (const auto* main = std::get_if<MainQsaProjection>(&p.input)) {
            const std::array qg{buffers.query.view({qw, m}), buffers.gate.view({qw, m})};
            ops::linear_split(input, main->query_gate, qg, arena, cuda);
        } else {
            const std::array all{buffers.key.view({kw, m}),  buffers.value.view({kw, m}),
                                 buffers.index_key,          buffers.query.view({qw, m}),
                                 buffers.gate.view({qw, m}), buffers.index_query.view({iq, m})};
            ops::linear_split(input, std::get<LinearParameters>(p.input), all, arena, cuda);
        }
    }
    if (const auto* main = std::get_if<MainQsaProjection>(&p.input)) {
        auto scratch = arena.scope();
        const std::array kv{buffers.key.view({kw, m}), buffers.value.view({kw, m}),
                            buffers.index_query.view({iq, m}), buffers.index_key};
        ops::linear_split(input, main->kv_index, kv, arena, cuda);
    }
    auto query       = buffers.query.view({d, qh, t, b});
    auto index_query = buffers.index_query.view({id, ih, t, b});
    const ops::QsaBatchView metadata{batch.positions, batch.valid_columns, batch.table_rows};
    ops::qsa_prepare({query, buffers.key.view({d, kh, t, b}), buffers.value.view({d, kh, t, b}),
                      index_query, buffers.index_key.view({ik, t, b})},
                     p.prepare, metadata, batch.rope_positions, before, cache, records, query,
                     index_query, cuda);
}

void extract(const ops::QsaTailStateView& before, ops::QsaTailStateView after,
             const ops::QsaRawRecordWriteView& records, const BatchView& batch,
             CudaExecutionView cuda) {
    ops::qsa_tail_extract(
        before,
        {records.raw_keys, records.rope_positions, batch.base_positions, batch.valid_columns},
        batch.valid_columns, after, cuda);
}
} // namespace

void qsa(const Tensor& input, const QsaConfig& c, const QsaParameters& p, const BatchView& batch,
         const ops::QsaTailStateView& before, std::optional<ops::QsaTailStateView> after,
         ops::QsaCacheView cache, ops::QsaRawRecordWriteView records, Tensor& output,
         MemoryArena& arena, CudaExecutionView cuda) {
    auto scope   = arena.scope();
    const auto t = batch.tokens, b = batch.batch;
    const auto d = dimension(c.head_dim), qh = dimension(c.query_heads);
    auto buffers = workspace::qsa(arena, c, t * b, !records.raw_keys.data);
    prepare(input, c, p, batch, before, cache, buffers, records, arena, cuda);
    const ops::QsaBatchView metadata{batch.positions, batch.valid_columns, batch.table_rows};
    ops::QsaBlockSelection selected{buffers.selection.view({dimension(c.block_budget), t, b}),
                                    buffers.counts.view({t, b})};
    {
        auto scratch = arena.scope();
        ops::qsa_indexer(
            buffers.index_query.view({dimension(c.index_dim), dimension(c.index_heads), t, b}),
            cache.index, metadata, {batch.max_visible_tokens}, arena, selected, cuda);
    }
    {
        auto scratch   = arena.scope();
        auto attention = buffers.output.view({d, qh, t, b});
        ops::qsa_sparse_gqa(buffers.query.view({d, qh, t, b}), buffers.gate.view({d, qh, t, b}),
                            cache.main, metadata, selected, {batch.max_visible_tokens}, arena,
                            attention, cuda);
    }
    ops::linear(buffers.output.view({dimension(c.query_width()), t * b}), p.output.weight, output,
                p.output.policy, arena, cuda);
    if (after) extract(before, *after, records, batch, cuda);
}

void qsa_history(const Tensor& input, const QsaConfig& c, const QsaHistoryParameters& p,
                 const BatchView& batch, const ops::QsaTailStateView& before,
                 ops::QsaTailStateView after, ops::QsaCacheView cache, MemoryArena& arena,
                 CudaExecutionView cuda) {
    auto scope   = arena.scope();
    const auto t = batch.tokens, b = batch.batch, m = t * b;
    const auto d = dimension(c.head_dim), kh = dimension(c.kv_heads);
    const auto kw = dimension(c.kv_width()), ik = dimension(c.index_key_width());
    auto buffers = workspace::qsa_history(arena, c, m);
    {
        auto scratch = arena.scope();
        const std::array outputs{buffers.key.view({kw, m}), buffers.value.view({kw, m}),
                                 buffers.index_key};
        ops::linear_split(input, p.input, outputs, arena, cuda);
    }
    const ops::QsaRawRecordWriteView records{buffers.raw_records.view({ik, t, b}),
                                             buffers.coordinate_records.view({3, t, b}),
                                             batch.record_offsets};
    ops::qsa_prepare_kv({buffers.key.view({d, kh, t, b}), buffers.value.view({d, kh, t, b}),
                         buffers.index_key.view({ik, t, b})},
                        p.prepare, {batch.positions, batch.valid_columns, batch.table_rows},
                        batch.rope_positions, before, cache, records, cuda);
    extract(before, after, records, batch, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
