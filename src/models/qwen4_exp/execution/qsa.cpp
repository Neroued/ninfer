#include "models/qwen4_exp/execution/qsa.h"
#include "models/qwen4_exp/execution/workspace.h"
#include "ninfer/ops/linear_split.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::execution {
namespace {

void extract(const ops::QsaTailStateView& before, ops::QsaTailStateView after,
             const ops::QsaRawRecordWriteView& records, const BatchView& batch,
             CudaExecutionView cuda) {
    ops::qsa_tail_extract(
        before,
        {records.raw_keys, records.rope_positions, batch.base_positions, batch.valid_columns},
        batch.valid_columns, after, batch.live_rows, cuda);
}
} // namespace

QsaRun::QsaRun(const Tensor& input, const QsaConfig& c, const QsaParameters& p, BatchView batch,
               ops::QsaTailStateView before, std::optional<ops::QsaTailStateView> after,
               ops::QsaCacheView cache, ops::QsaRawRecordWriteView records, Tensor output,
               MemoryArena& arena, CudaExecutionView cuda, const PackedPrefillState* packed)
    : input_(input), output_(output), config_(c), parameters_(p), batch_(batch), before_(before),
      after_(after), cache_(cache), records_(records), arena_(arena), cuda_(cuda), packed_(packed),
      scope_(arena.scope()), buffers_(workspace::qsa(arena, c, batch.tokens * batch.batch,
                                                     !packed && !records.raw_keys.data)) {}

void QsaRun::prepare() {
    const auto d = dimension(config_.head_dim), qh = dimension(config_.query_heads),
               kh = dimension(config_.kv_heads);
    const auto id = dimension(config_.index_dim), ih = dimension(config_.index_heads);
    const auto qw = dimension(config_.query_width()), kw = dimension(config_.kv_width());
    const auto iq = dimension(config_.index_query_width()),
               ik = dimension(config_.index_key_width());
    const auto t = batch_.tokens, b = batch_.batch, m = t * b;
    if (!packed_ && !records_.raw_keys.data)
        records_ = {buffers_.raw_records.view({ik, t, b}),
                    buffers_.coordinate_records.view({3, t, b}), batch_.record_offsets};
    {
        auto scratch = arena_.scope();
        if (const auto* main = std::get_if<MainQsaProjection>(&parameters_.input)) {
            const std::array qg{buffers_.query.view({qw, m}), buffers_.gate.view({qw, m})};
            ops::linear_split(input_, main->query_gate, qg, arena_, cuda_);
        } else {
            const std::array all{buffers_.key.view({kw, m}),  buffers_.value.view({kw, m}),
                                 buffers_.index_key,          buffers_.query.view({qw, m}),
                                 buffers_.gate.view({qw, m}), buffers_.index_query.view({iq, m})};
            ops::linear_split(input_, std::get<LinearParameters>(parameters_.input), all, arena_,
                              cuda_);
        }
    }
    if (const auto* main = std::get_if<MainQsaProjection>(&parameters_.input)) {
        auto scratch = arena_.scope();
        const std::array kv{buffers_.key.view({kw, m}), buffers_.value.view({kw, m}),
                            buffers_.index_query.view({iq, m}), buffers_.index_key};
        ops::linear_split(input_, main->kv_index, kv, arena_, cuda_);
    }
    auto query       = buffers_.query.view({d, qh, t, b});
    auto index_query = buffers_.index_query.view({id, ih, t, b});
    const ops::QsaProjectionView projected{query, buffers_.key.view({d, kh, t, b}),
                                           buffers_.value.view({d, kh, t, b}), index_query,
                                           buffers_.index_key.view({ik, t, b})};
    if (packed_) {
        auto scratch = arena_.scope();
        ops::qsa_prepare_update(
            projected, parameters_.prepare, {packed_->ragged, batch_.positions, batch_.table_rows},
            batch_.rope_positions, before_, cache_, *after_, query, index_query, arena_, cuda_);
        return;
    }
    const ops::QsaBatchView metadata{batch_.positions, batch_.valid_columns, batch_.table_rows};
    ops::qsa_prepare(projected, parameters_.prepare, metadata, batch_.rope_positions, before_,
                     cache_, records_, query, index_query, cuda_);
}

QsaReadout QsaRun::bind_readout() const {
    const int t = batch_.tokens, b = batch_.batch, d = dimension(config_.head_dim);
    const int qh = dimension(config_.query_heads);
    return {buffers_.query.view({d, qh, t, b}),
            buffers_.gate.view({d, qh, t, b}),
            buffers_.index_query.view(
                {dimension(config_.index_dim), dimension(config_.index_heads), t, b}),
            buffers_.output.view({d, qh, t, b}),
            cache_.index,
            cache_.main,
            {batch_.positions, batch_.valid_columns, batch_.table_rows},
            {buffers_.selection.view({dimension(config_.block_budget), t, b}),
             buffers_.counts.view({t, b})},
            {nullptr, 0, cuda_.device()},
            cuda_.multiprocessor_count()};
}

QsaReadout QsaRun::readout(ops::ContextBucketRange buckets) {
    auto call   = bind_readout();
    const int t = call.query.ne[2], b = call.query.ne[3];
    const auto bytes = std::max(ops::qsa_indexer_workspace_capacity_bytes(buckets, b, t, t),
                                ops::qsa_sparse_gqa_workspace_capacity_bytes(
                                    call.cache.storage, call.query.ne[1], call.cache.num_kv_heads,
                                    buckets, b, t, t, call.multiprocessor_count));
    if (bytes) {
        auto scope   = arena_.scope();
        call.scratch = arena_.alloc_bytes(bytes, 256);
    }
    return call;
}

ops::OpGraphProfile QsaReadout::index_profile(ops::ContextBucket bucket) const {
    return ops::qsa_indexer_graph_profile(bucket, query.ne[3], query.ne[2], multiprocessor_count);
}

ops::OpGraphProfile QsaReadout::attention_profile(ops::ContextBucket bucket, int sms) const {
    return ops::qsa_sparse_gqa_graph_profile(cache.storage, query.ne[1], cache.num_kv_heads, bucket,
                                             query.ne[3], query.ne[2], sms);
}

void QsaReadout::indexer(ops::ContextBucket bucket, CudaExecutionView cuda) const {
    MemoryArena arena(scratch);
    indexer(bucket, arena, cuda);
}

void QsaReadout::indexer(ops::ContextBucket bucket, MemoryArena& arena,
                         CudaExecutionView cuda) const {
    auto scope    = arena.scope();
    auto selected = selection;
    ops::qsa_indexer(index_query, index, batch, bucket, arena, selected, cuda);
}

void QsaReadout::attention(ops::ContextBucket bucket, CudaExecutionView cuda) const {
    MemoryArena arena(scratch);
    attention(bucket, arena, cuda);
}

void QsaReadout::attention(ops::ContextBucket bucket, MemoryArena& arena,
                           CudaExecutionView cuda) const {
    auto scope = arena.scope();
    auto out   = output;
    ops::qsa_sparse_gqa(query, gate, cache, batch, selection, bucket, arena, out, cuda);
}

void QsaReadout::execute(ops::ContextBucket bucket, MemoryArena& arena,
                         CudaExecutionView cuda) const {
    indexer(bucket, arena, cuda);
    attention(bucket, arena, cuda);
}

void QsaRun::finish() {
    ops::linear(
        buffers_.output.view({dimension(config_.query_width()), batch_.tokens * batch_.batch}),
        parameters_.output.weight, output_, parameters_.output.policy, arena_, cuda_);
    if (after_ && !packed_) extract(before_, *after_, records_, batch_, cuda_);
}

void QsaRun::execute(const QsaSubmit& submit) {
    prepare();
    if (packed_) {
        const auto bucket = ops::select_context_bucket(batch_.max_visible_tokens);
        const auto call   = bind_readout();
        const ops::QsaRaggedBatchView batch{packed_->ragged, batch_.positions, batch_.table_rows};
        auto selected = call.selection;
        auto output   = call.output;
        {
            auto scratch = arena_.scope();
            ops::qsa_indexer(call.index_query, call.index, batch, bucket, arena_, selected, cuda_);
        }
        {
            auto scratch = arena_.scope();
            ops::qsa_sparse_gqa(call.query, call.gate, call.cache, batch, selected, bucket, arena_,
                                output, cuda_);
        }
    } else if (submit) {
        const auto pages =
            static_cast<std::uint32_t>(cache_.main.k_pages.records.page_offsets.ne[0]);
        submit(readout(ops::ContextBucketRange::through(pages * kPagedKVPageSize)));
    } else {
        const auto bucket = ops::select_context_bucket(std::max(1U, batch_.max_visible_tokens));
        bind_readout().execute(bucket, arena_, cuda_);
    }
    finish();
}

void qsa_history(const Tensor& input, const QsaConfig& c, const QsaHistoryParameters& p,
                 const BatchView& batch, const ops::QsaTailStateView& before,
                 ops::QsaTailStateView after, ops::QsaCacheView cache, MemoryArena& arena,
                 CudaExecutionView cuda, const ops::RaggedSequenceView* ragged) {
    auto scope   = arena.scope();
    const auto t = batch.tokens, b = batch.batch, m = t * b;
    const auto d = dimension(c.head_dim), kh = dimension(c.kv_heads);
    const auto kw = dimension(c.kv_width()), ik = dimension(c.index_key_width());
    auto buffers = workspace::qsa_history(arena, c, m, !ragged);
    {
        auto scratch = arena.scope();
        const std::array outputs{buffers.key.view({kw, m}), buffers.value.view({kw, m}),
                                 buffers.index_key};
        ops::linear_split(input, p.input, outputs, arena, cuda);
    }
    const ops::QsaKvProjectionView projected{buffers.key.view({d, kh, t, b}),
                                             buffers.value.view({d, kh, t, b}),
                                             buffers.index_key.view({ik, t, b})};
    if (ragged) {
        ops::qsa_prepare_kv_update(projected, p.prepare,
                                   {*ragged, batch.positions, batch.table_rows},
                                   batch.rope_positions, before, cache, after, arena, cuda);
        return;
    }
    const ops::QsaRawRecordWriteView records{buffers.raw_records.view({ik, t, b}),
                                             buffers.coordinate_records.view({3, t, b}),
                                             batch.record_offsets};
    ops::qsa_prepare_kv(projected, p.prepare,
                        {batch.positions, batch.valid_columns, batch.table_rows},
                        batch.rope_positions, before, cache, records, cuda);
    extract(before, after, records, batch, cuda);
}

} // namespace ninfer::models::qwen4_exp::execution
