#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/state/continuation.h"
#include "models/qwen4_exp/execution/mtp.h"
#include "core/transfer.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/sequence.h"

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::commit_mtp(const Commit& commit) {
    const auto rows     = compute->rows;
    const auto& buffers = resources.buffers();
    const auto batch    = static_cast<std::int32_t>(compute->physical.rows);
    auto* accepted = static_cast<std::int32_t*>(host_tensor(buffers.batch.accepted_columns).data);
    auto* count_columns = static_cast<std::int32_t*>(host_tensor(buffers.batch.count_columns).data);
    auto* destinations =
        static_cast<std::int64_t*>(host_tensor(buffers.commit_destination_offsets).data);
    auto* last       = static_cast<std::int32_t*>(host_tensor(buffers.mtp->last_columns).data);
    auto* active     = static_cast<std::int32_t*>(host_tensor(buffers.mtp->active_columns).data);
    const auto width = compute->tokens;
    auto* skip       = static_cast<std::int32_t*>(host_tensor(buffers.mtp->skip_first).data);
    auto* bases = static_cast<std::int32_t*>(host_tensor(buffers.mtp->history_base_positions).data);
    bool any = false, rebuild_history = false, count_tokens = false;
    for (int b = 0; b < batch; ++b) {
        accepted[b] = count_columns[b] = 0;
        last[b] = active[b] = -1;
        destinations[b]     = -1;
        skip[b] = bases[b] = 0;
    }
    for (std::size_t b = 0; b < rows.size(); ++b) {
        const auto& decision = commit.rows[b];
        accepted[b]          = static_cast<std::int32_t>(decision.accepted);
        count_columns[b] = rows[b].sampling.presence_penalty || rows[b].sampling.frequency_penalty
                               ? decision.accepted
                               : 0;
        count_tokens |= count_columns[b] != 0;
        last[b]         = accepted[b] - 1;
        active[b]       = accepted[b] ? 0 : -1;
        destinations[b] = decision.states.at(parameters.profile->local.rank).destination;
        any |= decision.accepted != 0;
        const auto f     = rows[b].main_frontier;
        const auto pairs = mtp_history_frontier(f + decision.accepted) - rows[b].mtp_frontier;
        const std::uint32_t reused =
            f && decision.accepted && compute->verify && compute->verify->extents[b] > 0;
        skip[b]  = !f ? 1 : reused;
        bases[b] = rows[b].mtp_frontier + reused;
        rebuild_history |= pairs > reused;
    }
    if (!any) return;
    copy_bytes_async(
        buffers.commit_ingress.bind(resources.persistent()),
        resources.ingress().subspan(buffers.commit_ingress.offset - buffers.host_region.offset,
                                    buffers.commit_ingress.bytes),
        cuda);
    if (compute->graph)
        graphs->commit(*compute->graph_route, rebuild_history, count_tokens);
    else
        execute_mtp_commit(batch, width, rebuild_history, count_tokens);
}

void RankProgramImpl::execute_mtp_commit(std::int32_t batch, std::int32_t width,
                                         bool rebuild_history, bool count_tokens,
                                         const SectionRecorder& record_repair,
                                         const SectionRecorder& record_counts) {
    const auto& io      = resources.buffers();
    const auto& mtp     = *io.mtp;
    const auto k        = std::max(1, width - 1);
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto hidden   = dimension(parameters.profile->config.text.hidden);
    const auto row = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    auto accepted  = row(io.batch.accepted_columns);
    auto main      = resources.main_batch(width, batch, shape.max_context);
    main.destination_offsets = row(io.commit_destination_offsets);
    auto history       = resources.mtp_batch(history_control_view(mtp.history, k, batch), k, batch,
                                             shape.max_context);
    history.table_rows = row(mtp.draft_steps[0].table_rows);
    history.base_positions = row(mtp.history_base_positions);
    history.record_offsets = main.record_offsets;
    history.live_rows      = main.live_rows;
    auto last = row(mtp.last_columns), active = row(mtp.active_columns);
    const auto sample       = packed_tensor(io.pending_sample, {hidden, width, batch});
    const auto conditioning = packed_tensor(*io.pending_conditioning, {channels, width, batch});
    auto aligned            = packed_tensor(mtp.alignment_conditioning, {channels, k, batch});
    if (width > 1) {
        const auto records = resources.main_records(width, batch);
        resources.replay(width).fold.execute(main.source_offsets, main.destination_offsets,
                                             accepted, cuda);
        ops::qsa_tail_extract(
            resources.states->main_tail(main.source_offsets),
            {records.keys, records.coordinates, main.base_positions, main.valid_columns}, accepted,
            resources.states->main_tail(main.destination_offsets), main.live_rows, cuda);
        ops::ple_state_extract(
            resources.states->ple(main.source_offsets), {records.ple, main.valid_columns}, accepted,
            resources.states->ple(main.destination_offsets), main.live_rows, cuda);
    }
    const auto tail       = mtp_working_tail(0, batch);
    const auto key_width  = dimension(parameters.profile->config.text.qsa.index_key_width());
    const auto tail_width = dimension(parameters.profile->config.text.qsa.compression - 1);
    const auto repair     = [&] {
        ops::prepare_causal_pairs({conditioning,
                                   packed_tensor(mtp.base_conditioning, {channels, batch}),
                                   main.ids,
                                   main.positions,
                                   main.rope_positions.view({width, 3, batch}),
                                   packed_tensor(mtp.previous_rope, {3, batch}),
                                   accepted,
                                   row(mtp.skip_first),
                                   {}},
                                      {aligned, history.ids, history.positions, history.valid_columns,
                                   history.rope_positions.view({k, 3, batch})},
                                      cuda);

        execution::mtp_history(parameters.profile->config.text, parameters.profile->local.qsa,
                                   *parameters.mtp, {aligned, TokenPartition(k * batch)}, history, tail,
                                   tail, resources.mtp->layer(0), {}, resources.workspace, cuda);
    };
    if (record_repair)
        record_repair(repair);
    else if (rebuild_history)
        repair();
    auto destination     = resources.states->mtp_tail(main.destination_offsets);
    const auto tail_slot = 0U;
    auto source_keys     = packed_tensor(mtp.tail_keys, {key_width, tail_width, batch, 3})
                           .slice(3, tail_slot, 1)
                           .view({key_width * tail_width, 1, batch});
    auto key_store               = destination.raw_keys;
    key_store.field.tensor.shape = {key_width * tail_width, 1, 1, 1};
    const auto coordinates       = packed_tensor(mtp.tail_coordinates, {3, batch, 3})
                                 .slice(2, tail_slot, 1)
                                 .view({3, 1, batch});
    const std::array fields{
        ops::StateColumnScatter{sample, last,
                                resources.states->sample_hidden(main.destination_offsets)},
        ops::StateColumnScatter{conditioning, last,
                                resources.states->conditioning(main.destination_offsets)},
        ops::StateColumnScatter{source_keys, active, key_store},
        ops::StateColumnScatter{coordinates, active, destination.first_rope}};
    ops::scatter_state_fields(fields, cuda);
    const auto update_counts = [&] {
        auto counts = device_tensor(*io.token_counts);
        ops::increment_token_counts_batch(packed_tensor(*io.sampled_ids, {width, batch}),
                                          row(io.batch.count_columns), row(io.batch.count_slots),
                                          counts, cuda);
    };
    if (record_counts)
        record_counts(update_counts);
    else if (count_tokens)
        update_counts();
}

} // namespace ninfer::models::qwen4_exp::detail
