#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/execution/mtp.h"
#include "core/transfer.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/sequence.h"

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::commit_verify(const Commit& commit) {
    const auto rows     = compute->rows;
    const auto& buffers = resources.plan.persistent.buffers;
    const auto batch    = static_cast<std::int32_t>(rows.size());
    auto* accepted = static_cast<std::int32_t*>(host_tensor(buffers.batch.accepted_columns).data);
    bool any = false, rebuild_history = false, moved_destination = false;
    for (int b = 0; b < batch; ++b) {
        const auto& decision = commit.rows[b];
        accepted[b]          = static_cast<std::int32_t>(decision.accepted);
        any |= decision.accepted != 0;
        rebuild_history |= decision.accepted > 1;
        moved_destination |= decision.states.at(parameters.profile->local.rank).destination !=
                             rows[b].states.at(parameters.profile->local.rank).destination;
    }
    if (!any) return;

    auto accept = device_tensor(buffers.batch.accepted_columns).slice(0, 0, batch);
    copy_bytes_async({accept.data, accept.bytes(), cuda.device()},
                     {accepted, accept.bytes(), Device::cpu(), true}, cuda);
    if (moved_destination) {
        auto destinations = device_tensor(buffers.batch.destination_offsets).slice(0, 0, batch);
        auto host         = host_tensor(buffers.batch.destination_offsets).slice(0, 0, batch);
        auto* offsets     = static_cast<std::int64_t*>(host.data);
        for (int b = 0; b < batch; ++b)
            offsets[b] = commit.rows[b].states.at(parameters.profile->local.rank).destination;
        copy_bytes_async({destinations.data, destinations.bytes(), cuda.device()},
                         {host.data, host.bytes(), Device::cpu(), true}, cuda);
    }
    if (commit_graphs)
        commit_graphs->replay(batch, rebuild_history);
    else
        execute_mtp_commit(batch, rebuild_history);
}

void RankProgramImpl::execute_mtp_commit(std::int32_t batch, bool rebuild_history,
                                         DecodeGraphSection* captured_repair) {
    const auto& io   = resources.plan.persistent.buffers;
    const auto& mtp  = *io.mtp;
    const auto width = static_cast<int>(shape.draft_tokens + 1), k = width - 1;
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto hidden   = dimension(parameters.profile->config.text.hidden);
    const auto row     = [&](const TensorRegion& r) { return device_tensor(r).slice(0, 0, batch); };
    auto accepted      = row(io.batch.accepted_columns);
    const auto main    = resources.main_batch(width, batch, shape.max_context);
    auto history       = resources.mtp_batch(mtp.history, k, batch, shape.max_context);
    history.table_rows = row(mtp.draft_steps[0].table_rows);
    history.base_positions = main.base_positions;
    history.record_offsets = main.record_offsets;
    auto last = row(mtp.last_columns), active = row(mtp.active_columns);
    ops::prepare_shifted_prefix(main.ids, main.positions, main.rope_positions, accepted,
                                history.ids, history.positions, history.rope_positions,
                                history.valid_columns, last, active, cuda);
    resources.gdn_fold->execute(main.source_offsets, main.destination_offsets, accepted, cuda);
    for (std::uint32_t layer = 0; layer < parameters.profile->config.text.qsa_layers; ++layer) {
        const auto records = device_tensor(io.main_qsa.keys).slice(3, layer, 1).slice(2, 0, batch);
        const auto coordinates =
            device_tensor(io.main_qsa.coordinates).slice(3, layer, 1).slice(2, 0, batch);
        ops::qsa_tail_extract(resources.states->main_tail(layer, main.source_offsets),
                              {records, coordinates, main.base_positions, main.valid_columns},
                              accepted,
                              resources.states->main_tail(layer, main.destination_offsets), cuda);
    }
    ops::ple_state_extract(resources.states->ple(main.source_offsets),
                           {device_tensor(io.ple_records).slice(2, 0, batch), main.valid_columns},
                           accepted, resources.states->ple(main.destination_offsets), cuda);
    auto sample_store       = resources.states->sample_hidden(main.destination_offsets);
    auto conditioning_store = resources.states->conditioning(main.destination_offsets);
    const auto sample       = packed_tensor(io.pending_sample, {hidden, width, batch});
    const auto conditioning = packed_tensor(*io.pending_conditioning, {channels, width, batch});
    ops::scatter_batch_columns(sample, last, sample_store, cuda);
    ops::scatter_batch_columns(conditioning, last, conditioning_store, cuda);
    auto counts = device_tensor(*io.token_counts);
    ops::increment_token_counts_batch(packed_tensor(*io.sampled_ids, {width, batch}), accepted,
                                      row(io.batch.count_slots), counts, cuda);
    const auto tail       = mtp_working_tail(0, batch);
    const auto key_width  = dimension(parameters.profile->config.text.qsa.index_key_width());
    const auto tail_width = dimension(parameters.profile->config.text.qsa.compression - 1);
    // The repair destination starts with the candidate tail, including rows with zero history.
    const auto seed = [&](const TensorRegion& region, int features) {
        auto planes = packed_tensor(region, {features, batch, 3});
        auto source = planes.slice(2, 0, 1), destination = planes.slice(2, 1, 1);
        copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                         {source.data, source.bytes(), cuda.device()}, cuda);
    };
    seed(mtp.tail_keys, key_width * tail_width);
    seed(mtp.tail_coordinates, 3);
    const auto repair = [&] {
        auto aligned = packed_tensor(mtp.alignment_conditioning, {channels, k, batch});
        copy_rows_async({aligned.data, aligned.bytes(), cuda.device()}, channels * k * 2ULL,
                        {conditioning.data, conditioning.bytes(), cuda.device()},
                        channels * width * 2ULL, channels * k * 2ULL, batch, cuda);
        auto after = mtp_working_tail(1, batch);
        execution::mtp_history(parameters.profile->config.text, parameters.profile->local.qsa,
                               *parameters.mtp, aligned, history, tail, after,
                               resources.mtp->layer(0), {}, resources.workspace, cuda);
    };
    if (captured_repair)
        captured_repair->capture(cuda.stream(), repair);
    else if (rebuild_history)
        repair();
    auto destination     = resources.states->mtp_tail(main.destination_offsets);
    const auto tail_slot = 1U;
    auto source_keys     = packed_tensor(mtp.tail_keys, {key_width, tail_width, batch, 3})
                           .slice(3, tail_slot, 1)
                           .view({key_width * tail_width, 1, batch});
    auto key_store               = destination.raw_keys;
    key_store.field.tensor.shape = {key_width * tail_width, 1, 1, 1};
    ops::scatter_batch_columns(source_keys, active, key_store, cuda);
    const auto coordinates = packed_tensor(mtp.tail_coordinates, {3, batch, 3})
                                 .slice(2, tail_slot, 1)
                                 .view({3, 1, batch});
    ops::scatter_batch_columns(coordinates, active, destination.first_rope, cuda);
}

} // namespace ninfer::models::qwen4_exp::detail
