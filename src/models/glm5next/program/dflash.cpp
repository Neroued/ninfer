#include "models/glm5next/program/internal.h"

namespace ninfer::models::glm5next::detail {
Tensor RankProgramImpl::draft_features(const execution::BatchView& call) {
    const auto& field = frame.dflash->features;
    const auto& p     = *parameters.profile;
    const auto columns =
        call.sequence_owner
            ? execution::sequence_parallel_policy(p.local.size).partition(call.tokens, p.local.rank)
            : TokenPartition(call.tokens * call.rows);
    return resources.packed(field, {field.shape[0], columns.count / call.rows, call.rows});
}

execution::DFlashRun RankProgramImpl::dflash_run(int b, std::uint32_t bound) {
    const auto& io  = frame;
    const auto& d   = *io.dflash;
    const auto& s   = *io.speculative;
    const int steps = dimension(shape.draft_tokens), w = steps + 1;
    const int k = dimension(parameters.profile->config.draft->dflash2->selector_top_k);
    return execution::DFlashRun(
        parameters, {resources.workspace, resources.states, *parallel},
        {row(io.rows.anchors, b), row(io.rows.base_positions, b), row(d.valid_columns, b),
         row(io.rows.source_offsets, b), row(io.rows.live_rows, b),
         static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data), bound},
        {resources.packed(d.ids, {w, b}), resources.packed(d.positions, {w, b}),
         resources.packed(s.drafts, {steps, b}), resources.packed(d.candidate_ids, {k, steps, b}),
         resources.packed(d.proposal_q, {k, steps, b})},
        cuda);
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
    if (columns.split()) {
        auto gathered = resources.workspace.alloc(DType::BF16, {features.ne[0], retained});
        for (std::uint32_t rank = 0; rank < columns.ranks; ++rank) {
            const TokenPartition part(call.tokens, columns.ranks, rank);
            const int first = std::max(begin, part.begin);
            const int last  = std::min(call.tokens, part.begin + part.count);
            if (first >= last) continue;
            auto destination = gathered.slice(1, first - begin, last - first);
            if (rank == p.local.rank) {
                const auto source = features.slice(1, first - part.begin, last - first);
                copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                                 {source.data, source.bytes(), cuda.device()}, cuda);
            }
            parallel->broadcast(destination, rank, cuda);
        }
        features = gathered;
    }
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

void RankProgramImpl::accept_dflash(int b) {
    const auto& io  = frame;
    const auto& d   = *io.dflash;
    const int steps = dimension(shape.draft_tokens), w = steps + 1;
    const int k = dimension(parameters.profile->config.draft->dflash2->selector_top_k);
    auto logits =
        resources.packed(*io.logits, {dimension(parameters.profile->local.vocabulary.size), w, b});
    auto licensed = resources.packed(*io.sampled_ids, {w, b});
    auto counts   = row(*io.output_counts, b);
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data);
    vocabulary().accept_sparse(
        logits, configs, row(io.decode.valid_columns, b), row(io.rows.live_rows, b),
        draft_proposals(b), resources.packed(d.candidate_ids, {k, steps, b}),
        resources.packed(d.proposal_q, {k, steps, b}), row(io.speculative->extents, b),
        row(io.rows.sampling_positions, b), licensed, counts);
}
} // namespace ninfer::models::glm5next::detail
