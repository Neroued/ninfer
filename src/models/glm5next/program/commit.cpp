#include "models/glm5next/program/internal.h"

#include "core/transfer.h"
#include "models/glm5next/execution/mtp.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& commit) {
    prepare_commit(ticket, commit);
    auto& f = *compute;
    timer.resolve();
    f.release_vision    = std::any_of(commit.rows.begin(), commit.rows.end(),
                                      [](const auto& row) { return row.terminal; });
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (std::holds_alternative<MtpWork>(f.request.work.operation)) commit_mtp(commit);
    if (forward && forward->sample != SampleKind::None) {
        const auto& io = resources.plan.frame;
        const int b    = commit.rows.size();
        auto* values   = static_cast<std::int32_t*>(host(io.rows.count_columns).data);
        bool any       = false;
        for (int i = 0; i < b; ++i) {
            const auto logical   = f.physical.members[i];
            const auto& sampling = forward->rows[logical].sampling;
            values[i]            = sampling.presence_penalty || sampling.frequency_penalty
                                       ? commit.rows[logical].accepted
                                       : 0;
            any |= values[i] != 0;
        }
        if (any) {
            upload_region(io.rows.count_columns.region, b * sizeof(std::int32_t));
            auto counts = resources.tensor(*io.token_counts);
            ops::increment_token_counts_batch(
                resources.packed(*io.sampled_ids, {1, b}),
                resources.tensor(io.rows.count_columns).slice(0, 0, b),
                resources.tensor(io.rows.count_slots).slice(0, 0, b), counts, cuda);
        }
    }
    resources.compute_actions->submit(
        local_actions(commit.actions, parameters.profile->local.rank));
    timer.record_retirement();
    compute_done.record(cuda.stream());
    f.retirement = command;
    f.phase      = ComputePhase::Retiring;
}

void RankProgramImpl::commit_mtp(const Commit& commit) {
    auto& f        = *compute;
    const auto& io = resources.plan.frame;
    const int b    = f.batch.rows;
    auto* values   = static_cast<std::int32_t*>(host(io.rows.accepted_columns).data);
    auto* count_lengths = static_cast<std::int32_t*>(host(io.rows.count_columns).data);
    auto* last          = static_cast<std::int32_t*>(host(io.mtp->last_columns).data);
    bool any = false, repair = false, count_tokens = false;
    const auto& work = std::get<MtpWork>(f.request.work.operation);
    for (int i = 0; i < b; ++i) {
        const auto logical = f.physical.members[i];
        const auto n       = commit.rows[logical].accepted;
        values[i]          = n;
        const auto& sampling = work.rows[logical].sampling;
        count_lengths[i]     = sampling.presence_penalty || sampling.frequency_penalty ? n : 0;
        count_tokens |= count_lengths[i] != 0;
        last[i] = static_cast<std::int32_t>(n) - 1;
        any |= n != 0;
        repair |= work.rows[logical].attention_rank == parameters.profile->local.rank &&
                  n > (work.draft_extents[logical] ? 1U : 0U);
    }
    if (!any) return;
    const auto& a      = io.attention;
    const auto own     = f.physical.owners[parameters.profile->local.rank];
    auto* local_counts = static_cast<std::int32_t*>(host(a.accepted_columns).data);
    auto* local_active = static_cast<std::int32_t*>(host(a.active_columns).data);
    for (std::uint32_t j = 0; j < f.batch.attention.rows; ++j) {
        local_counts[j] = j < own.size ? values[own.begin + j] : 0;
        local_active[j] = local_counts[j] ? 0 : -1;
    }
    upload_region(io.commit_ingress, io.commit_ingress.bytes);
    if (f.graph_round)
        graphs->commit(b, f.batch.attention.rows, repair, count_tokens);
    else
        execute_mtp_commit(f.batch, repair, count_tokens);
}

void RankProgramImpl::execute_mtp_commit(const execution::BatchView& call, bool repair,
                                         bool count_tokens, DecodeGraphSection* captured_repair,
                                         DecodeGraphSection* captured_counts) {
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
    resources.kda_fold->execute(call.source_offsets, call.destination_offsets, accepted, cuda);
    const auto& local = call.attention;
    ops::kpool_tail_extract(resources.states.main_tail(local.source_offsets),
                            {resources.tensor(io.main_records.raw_keys).slice(2, 0, c),
                             resources.tensor(io.main_records.raw_gates).slice(2, 0, c),
                             row(io.attention.base_positions, c), local.valid_columns},
                            row(io.attention.accepted_columns, c),
                            resources.states.main_tail(local.destination_offsets), cuda);
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
        ops::StateColumnScatter{hidden, row(m.last_columns, b),
                                resources.states.hidden(call.destination_offsets)},
        ops::StateColumnScatter{scratch(m.tail_keys), row(io.attention.active_columns, c),
                                tail.raw_keys},
        ops::StateColumnScatter{scratch(m.tail_gates), row(io.attention.active_columns, c),
                                tail.raw_gates}};
    ops::scatter_state_fields(fields, cuda);
    const auto update_counts = [&] {
        auto counts = resources.tensor(*io.token_counts);
        ops::increment_token_counts_batch(resources.packed(*io.sampled_ids, {w, b}),
                                          row(io.rows.count_columns, b),
                                          row(io.rows.count_slots, b), counts, cuda);
    };
    if (captured_counts)
        captured_counts->capture(cuda.stream(), update_counts);
    else if (count_tokens)
        update_counts();
}

void RankProgramImpl::control_counts() {
    auto& f            = *compute;
    const auto& rows   = std::get<ForwardWork>(f.request.work.operation).rows;
    const auto& io     = resources.plan.frame;
    const auto cap     = resources.tensor(io.main.ids).numel();
    auto* ids          = static_cast<TokenId*>(host(io.main.ids).data);
    std::uint32_t used = 0;
    while (f.control_row < rows.size() && used < cap) {
        const auto& row = rows[f.control_row];
        const auto n =
            std::min<std::size_t>(cap - used, row.count_tokens.size() - f.control_offset);
        if (n) {
            std::copy_n(row.count_tokens.begin() + f.control_offset, n, ids + used);
            auto input = resources.tensor(io.main.ids).slice(0, used, n);
            copy_bytes_async({input.data, input.bytes(), cuda.device()},
                             {ids + used, input.bytes(), Device::cpu(), true}, cuda);
            auto counts =
                resources.tensor(*io.token_counts).slice(1, row.member.residency.lane.value, 1);
            ops::increment_token_counts(input, counts, cuda);
            f.control_offset += n;
            used += n;
        }
        if (f.control_offset == row.count_tokens.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
    }
    f.counts_pending = f.control_row < rows.size();
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}
} // namespace ninfer::models::glm5next::detail
