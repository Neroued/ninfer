#include "models/glm5next/program/internal.h"

#include "core/transfer.h"
#include "models/glm5next/state/continuation.h"
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
    if (std::holds_alternative<SpeculativeWork>(f.request.work.operation) ||
        (forward && forward->speculative))
        commit_speculative(commit);
    else if (forward && forward->sample != SampleKind::None) {
        const auto& io     = frame;
        const auto members = f.sampling_members();
        const int b        = extent(members.size());
        auto* values       = static_cast<std::int32_t*>(host(io.rows.count_columns).data);
        bool any           = false;
        for (int i = 0; i < b; ++i) {
            const auto logical   = members[i];
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

void RankProgramImpl::commit_speculative(const Commit& commit) {
    auto& f             = *compute;
    const auto& io      = frame;
    const int b         = f.batch.rows;
    auto* values        = static_cast<std::int32_t*>(host(io.rows.accepted_columns).data);
    auto* count_lengths = static_cast<std::int32_t*>(host(io.rows.count_columns).data);
    auto* last          = static_cast<std::int32_t*>(host(io.speculative->last_columns).data);
    bool any = false, repair = false, count_tokens = false;
    const auto rows      = work_rows(f.request.work);
    const auto* proposal = std::get_if<SpeculativeWork>(&f.request.work.operation);
    auto* skip = parameters.mtp ? static_cast<int*>(host(io.mtp->skip_first).data) : nullptr;
    if (skip) std::fill_n(skip, b, 0);
    std::fill_n(values, b, 0);
    std::fill_n(count_lengths, b, 0);
    std::fill_n(last, b, -1);
    for (std::size_t i = 0; i < f.physical.members.size(); ++i) {
        const auto logical   = f.physical.members[i];
        const auto n         = commit.rows[logical].accepted;
        values[i]            = n;
        const auto& sampling = rows[logical].sampling;
        count_lengths[i]     = sampling.presence_penalty || sampling.frequency_penalty ? n : 0;
        count_tokens |= count_lengths[i] != 0;
        last[i] = static_cast<std::int32_t>(n) - 1;
        any |= n != 0;
        if (skip) {
            const auto& row = rows[logical];
            const auto m    = mtp_history_frontier(row.main_frontier + n) - row.mtp_frontier;
            const bool reuse =
                n && row.main_frontier && proposal && proposal->draft_extents[logical];
            skip[i] = !row.main_frontier || reuse;
            repair |= row.attention_rank == parameters.profile->local.rank && m > unsigned(reuse);
        }
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
    if (f.graph_route)
        graphs->commit(*f.graph_route, parameters.mtp && repair, count_tokens);
    else
        execute_speculative_commit(f.batch, repair, count_tokens);
}

void RankProgramImpl::execute_speculative_commit(const execution::BatchView& call, bool repair,
                                                 bool count_tokens,
                                                 const SectionRecorder& record_repair,
                                                 const SectionRecorder& record_counts) {
    const auto& io = frame;
    const int b = call.rows, w = call.tokens, c = call.attention.rows;
    const int h   = dimension(parameters.profile->config.text.hidden);
    auto accepted = row(io.rows.accepted_columns, b);
    if (w > 1) {
        resources.replay(w).fold.execute(call.source_offsets, call.destination_offsets, accepted,
                                         cuda);
        const auto& local = call.attention;
        ops::kpool_tail_extract(resources.states.main_tail(local.source_offsets),
                                {resources.raw_record(io.main_records.raw_keys, w, c),
                                 resources.raw_record(io.main_records.raw_gates, w, c),
                                 row(io.attention.base_positions, c), local.valid_columns},
                                row(io.attention.accepted_columns, c),
                                resources.states.main_tail(local.destination_offsets), cuda);
    }
    if (parameters.mtp) {
        repair_mtp_history(call, repair, record_repair);
    } else {
        append_dflash_context(call, accepted, {0, std::uint32_t(w)});
        const std::array fields{ops::StateColumnScatter{
            resources.packed(io.pending_hidden, {h, w, b}), row(io.speculative->last_columns, b),
            resources.states.hidden(call.destination_offsets)}};
        ops::scatter_state_fields(fields, cuda);
    }
    const auto update_counts = [&] {
        auto counts = resources.tensor(*io.token_counts);
        ops::increment_token_counts_batch(resources.packed(*io.sampled_ids, {w, b}),
                                          row(io.rows.count_columns, b),
                                          row(io.rows.count_slots, b), counts, cuda);
    };
    if (record_counts)
        record_counts(update_counts);
    else if (count_tokens)
        update_counts();
}

void RankProgramImpl::control_counts() {
    auto& f            = *compute;
    const auto& rows   = std::get<ForwardWork>(f.request.work.operation).rows;
    const auto& io     = frame;
    const auto cap     = resources.tensor(io.main.ids).numel();
    auto* ids          = static_cast<TokenId*>(host(io.main.ids).data);
    std::uint32_t used = 0;
    while (f.control_row < rows.size() && used < cap) {
        const auto& row = rows[f.control_row];
        const auto n    = std::min<std::size_t>(cap - used, row.ids.size() - f.control_offset);
        if (n) {
            ControlTokens{row.ids, *row.control_last_token}.copy({ids + used, n}, f.control_offset);
            auto input = resources.tensor(io.main.ids).slice(0, used, n);
            copy_bytes_async({input.data, input.bytes(), cuda.device()},
                             {ids + used, input.bytes(), Device::cpu(), true}, cuda);
            auto counts =
                resources.tensor(*io.token_counts).slice(1, row.member.residency.lane.value, 1);
            ops::increment_token_counts(input, counts, cuda);
            f.control_offset += n;
            used += n;
        }
        if (f.control_offset == row.ids.size()) {
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
