#include "models/glm5next/program/internal.h"

#include "core/nvtx.h"

namespace ninfer::models::glm5next::detail {
namespace {
Tensor gather_feature_range(const Tensor& features, const TokenPartition& columns, int begin,
                            int end, MemoryArena& workspace, ParallelExecution& parallel,
                            CudaExecutionView cuda) {
    auto gathered = workspace.alloc(DType::BF16, {features.ne[0], end - begin});
    for (std::uint32_t rank = 0; rank < columns.ranks; ++rank) {
        const TokenPartition part(columns.columns, columns.ranks, rank);
        const int first = std::max(begin, part.begin);
        const int last  = std::min(end, part.begin + part.count);
        if (first >= last) continue;
        auto destination = gathered.slice(1, first - begin, last - first);
        if (rank == columns.rank) {
            const auto source = features.slice(1, first - part.begin, last - first);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {source.data, source.bytes(), cuda.device()}, cuda);
        }
        parallel.broadcast(destination, rank, cuda);
    }
    return gathered;
}
} // namespace

Tensor RankProgramImpl::draft_features(const execution::BatchView& call, bool packed) {
    const auto& field = frame.dflash->features;
    const auto& p     = *parameters.profile;
    const auto columns =
        call.sequence_owner || packed
            ? execution::sequence_parallel_policy(p.local.size).partition(call.tokens, p.local.rank)
            : TokenPartition(call.tokens * call.rows);
    return resources.packed(field, {field.shape[0], columns.count / call.rows, call.rows});
}

execution::DFlashRun RankProgramImpl::dflash_run(int b, ops::ContextBucket bucket, int steps) {
    const auto& io = frame;
    const auto& d  = *io.dflash;
    const auto& s  = *io.speculative;
    const int w    = dimension(shape.draft_tokens + 1);
    const int k    = dimension(parameters.profile->config.draft->dflash2->selector_top_k);
    return execution::DFlashRun(
        parameters, {resources.workspace, resources.states, *parallel},
        {row(io.rows.anchors, b), row(io.rows.base_positions, b), row(d.valid_columns, b),
         row(io.rows.source_offsets, b), row(d.proposal_rows, b),
         static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data), bucket},
        {resources.packed(d.ids, {w, b}), resources.packed(d.positions, {w, b}),
         resources.packed(s.drafts, {steps, b}), resources.packed(d.candidate_ids, {k, steps, b}),
         resources.packed(d.proposal_q, {k, steps, b}),
         resources.packed(*io.draft_confidences, {steps, b})},
        cuda);
}

void RankProgramImpl::dflash_forward(int rows, ops::ContextBucket bucket, int head_columns,
                                     const execution::DraftSubmit& submit) {
    nvtx::ScopedRange range(nvtx::Name::DFlashProposal, nvtx::Category::DFlash);
    auto run = dflash_run(rows, bucket, head_columns);
    run.stem();
    for (std::uint32_t layer = 0; layer < parameters.draft->layers.size(); ++layer) {
        run.attention(layer).execute(submit);
        run.mlp(layer);
    }
    run.finish();
}

void RankProgramImpl::append_dflash_context(const execution::BatchView& call, const Tensor& counts,
                                            ops::ContextKVMaterializeExecutionEnvelope envelope) {
    auto scope          = resources.workspace.scope();
    auto features       = draft_features(call);
    auto positions      = call.positions;
    auto live           = counts;
    const auto& p       = *parameters.profile;
    const auto capacity = p.draft->context.capacity;
    const auto columns =
        call.sequence_owner
            ? execution::sequence_parallel_policy(p.local.size).partition(call.tokens, p.local.rank)
            : TokenPartition(call.tokens * call.rows);
    const int retained = std::min(std::uint32_t(call.tokens), capacity);
    const int begin    = call.tokens - retained;
    if (columns.split())
        features = gather_feature_range(features, columns, begin, call.tokens, resources.workspace,
                                        *parallel, cuda);
    if (std::uint32_t(call.tokens) > capacity) {
        if (call.rows != 1 || envelope.min_count != std::uint32_t(call.tokens) ||
            envelope.max_count != std::uint32_t(call.tokens))
            throw std::logic_error(
                "DFlash window replacement requires one fully committed sequence");
        if (!columns.split()) features = features.slice(1, begin, capacity);
        positions = positions.slice(0, begin, capacity);
        live      = resources.tensor(frame.dflash->window_count);
        envelope  = {capacity, capacity};
    }
    execution::draft_append_context(parameters, {resources.workspace, resources.states, *parallel},
                                    features, positions, live, call.destination_offsets, envelope,
                                    cuda);
}

void RankProgramImpl::append_packed_dflash_context(const execution::BatchView& call,
                                                   const ops::RaggedSequenceView& ragged) {
    const auto& p = *parameters.profile;
    const auto columns =
        execution::sequence_parallel_policy(p.local.size).partition(call.tokens, p.local.rank);
    const auto features = draft_features(call, true);
    const auto capacity = p.draft->context.capacity;
    // Draft context is TP-sharded, so both ranks materialize every segment in the same order.
    for (int s = 0; s < ragged.num_seqs(); ++s) {
        const auto len      = std::uint32_t(ragged.segment_length(s));
        const auto retained = std::min(len, capacity);
        const int count     = extent(retained);
        if (!count) continue;
        auto scope           = resources.workspace.scope();
        const int end        = ragged.host_cu_seqlens[s + 1];
        const int begin      = end - count;
        const auto tail      = columns.split()
                                   ? gather_feature_range(features, columns, begin, end,
                                                          resources.workspace, *parallel, cuda)
                                   : features.slice(1, begin, count);
        const auto positions = call.positions.slice(0, begin, count).view({count, 1});
        const auto counts    = len > capacity ? resources.tensor(frame.dflash->window_count)
                                              : ragged.seq_lengths.slice(0, s, 1);
        execution::draft_append_context(
            parameters, {resources.workspace, resources.states, *parallel}, tail, positions, counts,
            call.destination_offsets.slice(0, s, 1), {retained, retained}, cuda);
    }
}

void RankProgramImpl::accept_dflash(int b, int w, int steps) {
    const auto& io = frame;
    const auto& d  = *io.dflash;
    const int k    = dimension(parameters.profile->config.draft->dflash2->selector_top_k);
    auto logits =
        resources.packed(*io.logits, {dimension(parameters.profile->local.vocabulary.size), w, b});
    auto licensed = resources.packed(*io.sampled_ids, {w, b});
    auto counts   = row(*io.output_counts, b);
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data);
    const auto verified = std::min(steps, w - 1);
    vocabulary().accept_sparse(
        logits, configs, row(io.decode.valid_columns, b), row(io.rows.live_rows, b),
        draft_proposals(b, steps).slice(0, 0, verified),
        resources.packed(d.candidate_ids, {k, steps, b}).slice(1, 0, verified),
        resources.packed(d.proposal_q, {k, steps, b}).slice(1, 0, verified),
        row(io.speculative->extents, b), row(io.rows.sampling_positions, b), licensed, counts);
}
} // namespace ninfer::models::glm5next::detail
