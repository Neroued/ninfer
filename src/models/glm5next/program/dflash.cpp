#include "models/glm5next/program/internal.h"

namespace ninfer::models::glm5next::detail {
Tensor RankProgramImpl::draft_features(int tokens, int rows) {
    const auto& field = resources.plan.frame.dflash->features;
    return resources.packed(field, {field.shape[0], tokens, rows});
}

execution::DFlashRun RankProgramImpl::dflash_run(int b, std::uint32_t bound) {
    const auto& io  = resources.plan.frame;
    const auto& d   = *io.dflash;
    const auto& s   = *io.speculative;
    const int steps = dimension(shape.draft_tokens), w = steps + 1;
    const int k = dimension(parameters.profile->config.draft->dflash2->selector_top_k);
    return execution::DFlashRun(
        parameters, {resources.workspace, resources.states, *parallel},
        {row(io.rows.anchors, b), row(io.rows.base_positions, b), row(d.valid_columns, b),
         row(io.rows.source_offsets, b),
         static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data), bound},
        {resources.packed(d.ids, {w, b}), resources.packed(d.positions, {w, b}),
         resources.packed(s.drafts, {steps, b}), resources.packed(d.candidate_ids, {k, steps, b}),
         resources.packed(d.proposal_q, {k, steps, b})},
        cuda);
}

void RankProgramImpl::append_dflash_context(const execution::BatchView& call, const Tensor& counts,
                                            ops::ContextKVMaterializeExecutionEnvelope envelope) {
    auto features       = draft_features(call.tokens, call.rows);
    auto positions      = call.positions;
    auto live           = counts;
    const auto capacity = parameters.profile->draft->context.capacity;
    if (std::uint32_t(call.tokens) > capacity) {
        if (call.rows != 1 || envelope.min_count != std::uint32_t(call.tokens) ||
            envelope.max_count != std::uint32_t(call.tokens))
            throw std::logic_error(
                "DFlash window replacement requires one fully committed sequence");
        const auto begin = call.tokens - capacity;
        features         = features.slice(1, begin, capacity);
        positions        = positions.slice(0, begin, capacity);
        live             = resources.tensor(resources.plan.frame.dflash->window_count);
        envelope         = {capacity, capacity};
    }
    execution::draft_append_context(parameters, {resources.workspace, resources.states, *parallel},
                                    features, positions, live, call.destination_offsets, envelope,
                                    cuda);
}

void RankProgramImpl::accept_dflash(int b) {
    const auto& io  = resources.plan.frame;
    const auto& d   = *io.dflash;
    const int steps = dimension(shape.draft_tokens), w = steps + 1;
    const int k = dimension(parameters.profile->config.draft->dflash2->selector_top_k);
    auto logits =
        resources.packed(*io.logits, {dimension(parameters.profile->local.vocabulary.size), w, b});
    auto licensed = resources.packed(*io.sampled_ids, {w, b});
    auto counts   = row(*io.output_counts, b);
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data);
    vocabulary().accept_sparse(logits, configs, row(io.decode.valid_columns, b), draft_proposals(b),
                               resources.packed(d.candidate_ids, {k, steps, b}),
                               resources.packed(d.proposal_q, {k, steps, b}),
                               row(io.speculative->extents, b), row(io.rows.sampling_positions, b),
                               licensed, counts);
}
} // namespace ninfer::models::glm5next::detail
