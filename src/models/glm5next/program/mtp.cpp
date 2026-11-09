#include "models/glm5next/program/internal.h"

#include "models/glm5next/execution/mtp.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/sequence.h"
#include "ninfer/ops/speculative_accept.h"
#include "ninfer/ops/vocabulary.h"

namespace ninfer::models::glm5next::detail {
void RankProgramImpl::stage_conditioning(int rows) {
    const auto& io = resources.plan.frame;
    auto out       = resources.packed(io.mtp->previous_hidden,
                                      {dimension(parameters.profile->config.text.hidden), rows});
    ops::gather(resources.states.hidden(row(io.rows.source_offsets, rows)),
                row(io.rows.base_positions, rows), out, cuda);
}

void RankProgramImpl::gather_decode_state(const execution::BatchView& call) {
    const auto& io = resources.plan.frame;
    const auto& m  = *io.mtp;
    const int c = call.attention.rows, h = parameters.profile->config.text.hidden;
    const int d                      = m.tail_keys.shape[0] * m.tail_keys.shape[1];
    auto tail                        = resources.states.mtp_tail(call.attention.source_offsets);
    tail.raw_keys.field.tensor.shape = tail.raw_gates.field.tensor.shape = {d, 1, 1, 1};
    const auto scratch = [&](const TensorRegion& input) {
        return resources.tensor(input).slice(3, 0, 1).slice(2, 0, c).view({d, c});
    };
    const std::array fields{
        ops::StateGather{resources.states.hidden(call.source_offsets),
                         row(io.rows.base_positions, call.rows),
                         resources.packed(m.previous_hidden, {h, call.rows})},
        ops::StateGather{tail.raw_keys, call.attention.live_rows, scratch(m.tail_keys)},
        ops::StateGather{tail.raw_gates, call.attention.live_rows, scratch(m.tail_gates)}};
    ops::gather_state_fields(fields, cuda);
}

Tensor RankProgramImpl::mtp_output_ids(int step, int batch) {
    return resources.tensor(resources.plan.frame.speculative->drafts)
        .slice(1, step, 1)
        .slice(0, 0, batch)
        .view({batch});
}

void RankProgramImpl::prepare_history_metadata(std::span<const WorkRow> rows,
                                               std::span<const std::uint32_t> skip_first) {
    const auto& m = *resources.plan.frame.mtp;
    auto* values  = static_cast<std::int32_t*>(host(m.skip_first).data);
    for (std::size_t b = 0; b < rows.size(); ++b) {
        const auto logical = compute->physical.members[b];
        values[b] = skip_first.empty() ? rows[logical].main_frontier == 0 : skip_first[logical];
    }
    if (compute->sequence) upload_region(m.skip_first.region, rows.size() * sizeof(std::int32_t));
}

void RankProgramImpl::align_history(const execution::BatchView& main, Tensor hidden, int width) {
    const auto& m = *resources.plan.frame.mtp;
    const int b = main.rows, t = main.tokens, h = parameters.profile->config.text.hidden;
    const int c = main.attention.rows;
    auto call =
        batch(m.history, width, c, c, main.attention.max_visible_tokens, main.sequence_owner);
    auto conditioning = resources.packed(m.conditioning, {h, width, c});
    ops::prepare_causal_pairs({hidden.view({h, t, b}),
                               resources.packed(m.previous_hidden, {h, b}),
                               main.ids,
                               main.positions,
                               {},
                               {},
                               main.valid_columns,
                               row(m.skip_first, b),
                               main.attention.to_global},
                              {conditioning, call.ids, call.positions, call.valid_columns, {}},
                              cuda);
    const bool step = !main.sequence_owner;
    const auto before =
        step ? resources.mtp_tail(0, c) : resources.states.mtp_tail(main.attention.source_offsets);
    const auto after =
        step ? before : resources.states.mtp_tail(main.attention.destination_offsets);
    execution::mtp_history(parameters, {resources.workspace, *resources.mtp, before, after}, call,
                           conditioning.view({h, width * c}), cuda);
    if (step) {
        auto destination = resources.states.mtp_tail(main.attention.destination_offsets);
        const int d      = m.tail_keys.shape[0] * m.tail_keys.shape[1];
        destination.raw_keys.field.tensor.shape =
            destination.raw_gates.field.tensor.shape = {d, 1, 1, 1};
        const auto scratch                           = [&](const TensorRegion& input) {
            return resources.tensor(input).slice(3, 0, 1).slice(2, 0, c).view({d, 1, c});
        };
        const std::array fields{
            ops::StateColumnScatter{scratch(m.tail_keys),
                                    row(resources.plan.frame.attention.live_columns, c),
                                    destination.raw_keys},
            ops::StateColumnScatter{scratch(m.tail_gates),
                                    row(resources.plan.frame.attention.live_columns, c),
                                    destination.raw_gates}};
        ops::scatter_state_fields(fields, cuda);
    }
}

execution::MtpRun RankProgramImpl::mtp_run(std::uint32_t step, const execution::BatchView& call) {
    const int c       = call.attention.rows;
    const auto before = step ? resources.mtp_tail(step == 1 ? 0 : 1 + (step - 2) % 2, c)
                             : resources.states.mtp_tail(call.attention.source_offsets);
    const auto after  = resources.mtp_tail(step ? 1 + (step - 1) % 2 : 0, c);
    return execution::MtpRun(
        parameters,
        {resources.workspace, *resources.mtp, before, after, parallel ? &*parallel : nullptr}, call,
        cuda);
}

execution::MlaSelection RankProgramImpl::mtp_selection(std::uint32_t step, int c) {
    auto selected          = resources.plan.frame.mtp->selection.bind(resources.fixed());
    selected.pool_ids      = selected.pool_ids.slice(1, 0, c);
    selected.pool_counts   = selected.pool_counts.slice(0, 0, c);
    selected.selection_end = selected.selection_end.slice(0, 0, c);
    return {selected, step && parameters.profile->config.text.mla.share_mtp_selection};
}

void RankProgramImpl::mtp_forward(std::uint32_t step, const execution::BatchView& call,
                                  const execution::MlaSubmit& submit) {
    const auto& io = resources.plan.frame;
    const auto& m  = *io.mtp;
    const int b = call.rows, c = call.attention.rows, h = parameters.profile->config.text.hidden;
    auto recursive      = resources.packed(m.recursive_hidden, {h, b, 2});
    const auto previous = step ? recursive.slice(2, (step - 1) % 2, 1).view({h, b})
                               : resources.packed(m.previous_hidden, {h, b});
    auto next           = recursive.slice(2, step % 2, 1).view({h, b});
    {
        auto run = mtp_run(step, call);
        run.stem(previous);
        run.local_attention(mtp_selection(step, c), submit);
        run.finish_attention();
        run.ffn();
        run.finish(next);
    }
    auto logits =
        resources.packed(*io.logits, {dimension(parameters.profile->local.vocabulary.size), b});
    const auto& head = parameters.mtp->head;
    ops::linear(next, head.weight, logits, head.policy, resources.workspace, cuda);
    auto ids = mtp_output_ids(step, b);
    if (parallel) {
        auto keys = resources.packed(*io.candidates, {1, 1, b});
        const auto* configs =
            static_cast<const ops::SamplingConfig*>(resources.tensor(m.proposal_sampling).data);
        ops::vocabulary_candidates(logits.view({logits.ne[0], 1, b}),
                                   parameters.profile->local.vocabulary.begin,
                                   parameters.profile->public_tokens, configs,
                                   row(m.proposal_columns, b), {}, keys, resources.workspace, cuda);
        parallel->maximum(keys, cuda);
        ops::sample_vocabulary_candidates(keys.view({1, 1, b, 1}), configs,
                                          row(io.rows.sampling_positions, b),
                                          ops::kSamplePurposeDecode, ids, cuda);
    } else {
        ops::argmax(logits, ids, parameters.profile->public_tokens, cuda);
    }
}

void RankProgramImpl::accept_mtp(int b) {
    const auto& io    = resources.plan.frame;
    const auto& m     = *io.mtp;
    const auto& local = parameters.profile->local;
    const int w       = shape.draft_tokens + 1;
    auto logits       = resources.packed(*io.logits, {dimension(local.vocabulary.size), w, b});
    auto drafts       = draft_proposals(b);
    auto positions = row(io.rows.sampling_positions, b), extents = row(io.speculative->extents, b);
    auto licensed = resources.packed(*io.sampled_ids, {w, b});
    auto counts   = row(*io.output_counts, b);
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(resources.tensor(*io.sampling).data);
    if (!parallel) {
        auto target = resources.packed(*m.target_tokens, {w, b});
        auto flat   = target.view({w * b});
        ops::argmax(logits.view({logits.ne[0], w * b}), flat, parameters.profile->public_tokens,
                    cuda);
        ops::speculative_accept_greedy_drafts(target, logits, drafts, extents, positions, licensed,
                                              counts, parameters.profile->public_tokens, configs,
                                              resources.workspace, cuda);
        return;
    }
    vocabulary().accept(logits, configs, row(io.decode.valid_columns, b), drafts, extents,
                        positions, licensed, counts);
}

void RankProgramImpl::repair_mtp_history(const execution::BatchView& call, bool repair,
                                         DecodeGraphSection* captured_repair) {
    const auto& io = resources.plan.frame;
    const auto& m  = *io.mtp;
    const int b = call.rows, w = call.tokens, c = call.attention.rows,
              h       = parameters.profile->config.text.hidden;
    const auto hidden = resources.packed(io.pending_hidden, {h, w, b});
    auto accepted     = row(io.rows.accepted_columns, b);
    auto history      = batch(m.history, w - 1, c, c, call.attention.max_visible_tokens);
    auto conditioning = resources.packed(m.conditioning, {h, w - 1, c});
    ops::prepare_causal_pairs(
        {hidden,
         resources.packed(m.previous_hidden, {h, b}),
         call.ids,
         call.positions,
         {},
         {},
         accepted,
         row(m.skip_first, b),
         call.attention.to_global},
        {conditioning, history.ids, history.positions, history.valid_columns, {}}, cuda);
    const auto& local         = call.attention;
    const auto history_update = [&] {
        const auto tail = resources.mtp_tail(0, c);
        execution::mtp_history(parameters, {resources.workspace, *resources.mtp, tail, tail},
                               history, conditioning.view({h, (w - 1) * c}), cuda);
    };
    if (captured_repair)
        captured_repair->capture(cuda.stream(), history_update);
    else if (repair)
        history_update();
    auto tail                        = resources.states.mtp_tail(local.destination_offsets);
    const int features               = m.tail_keys.shape[0] * m.tail_keys.shape[1];
    tail.raw_keys.field.tensor.shape = tail.raw_gates.field.tensor.shape = {features, 1, 1, 1};
    const auto scratch = [&](const TensorRegion& input) {
        return resources.tensor(input).slice(3, 0, 1).slice(2, 0, c).view({features, 1, c});
    };
    const std::array fields{
        ops::StateColumnScatter{hidden, row(io.speculative->last_columns, b),
                                resources.states.hidden(call.destination_offsets)},
        ops::StateColumnScatter{scratch(m.tail_keys), row(io.attention.active_columns, c),
                                tail.raw_keys},
        ops::StateColumnScatter{scratch(m.tail_gates), row(io.attention.active_columns, c),
                                tail.raw_gates}};
    ops::scatter_state_fields(fields, cuda);
}

} // namespace ninfer::models::glm5next::detail
