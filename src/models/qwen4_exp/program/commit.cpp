#include "models/qwen4_exp/program/internal.h"
#include "models/qwen4_exp/execution/mtp.h"
#include "core/transfer.h"

#include <algorithm>
#include <array>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::commit_verify(const Commit& commit) {
    const auto& round   = compute->verify.value();
    const auto rows     = compute->rows;
    const auto& buffers = resources.plan.persistent.buffers;
    const auto& frame   = *buffers.mtp;
    const auto batch    = static_cast<std::int32_t>(rows.size());
    const auto width    = static_cast<std::int32_t>(round.width);
    const auto channels = dimension(parameters.profile->config.text.residual_width());
    const auto hidden   = dimension(parameters.profile->config.text.hidden);
    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold;
    auto* accepted = static_cast<std::int32_t*>(host_tensor(buffers.batch.accepted_columns).data);
    bool any = false, rebuild_history = false, moved_destination = false;
    for (int b = 0; b < batch; ++b) {
        const auto& decision = commit.rows[b];
        accepted[b]          = static_cast<std::int32_t>(decision.accepted);
        any |= decision.accepted != 0;
        rebuild_history |= decision.accepted > 1;
        moved_destination |= decision.slots.destination != rows[b].slots.destination;
        fold[b] = {decision.slots.source, decision.slots.destination, accepted[b]};
    }
    if (!any) return;

    auto accept = device_tensor(buffers.batch.accepted_columns).slice(0, 0, batch);
    copy_bytes_async({accept.data, accept.bytes(), cuda.device()},
                     {accepted, accept.bytes(), Device::cpu(), true}, cuda);
    if (moved_destination) {
        auto destinations = device_tensor(buffers.batch.destination_slots).slice(0, 0, batch);
        auto host         = host_tensor(buffers.batch.destination_slots).slice(0, 0, batch);
        auto* slots       = static_cast<std::int32_t*>(host.data);
        for (int b = 0; b < batch; ++b) slots[b] = commit.rows[b].slots.destination;
        copy_bytes_async({destinations.data, destinations.bytes(), cuda.device()},
                         {host.data, host.bytes(), Device::cpu(), true}, cuda);
    }
    resources.gdn_fold->execute({fold.data(), rows.size()}, cuda);
    const auto& metadata = round.batch;
    for (std::uint32_t layer = 0; layer < parameters.profile->config.text.qsa_layers; ++layer) {
        const auto records =
            device_tensor(buffers.main_qsa.keys).slice(3, layer, 1).slice(2, 0, batch);
        const auto coordinates =
            device_tensor(buffers.main_qsa.coordinates).slice(3, layer, 1).slice(2, 0, batch);
        ops::qsa_tail_extract(
            resources.states->main_tail(layer, metadata.source_slots),
            {records, coordinates, metadata.base_positions, metadata.valid_columns}, accept,
            resources.states->main_tail(layer, metadata.destination_slots), cuda);
    }
    ops::ple_state_extract(
        resources.states->ple(metadata.source_slots),
        {device_tensor(buffers.ple_records).slice(2, 0, batch), metadata.valid_columns}, accept,
        resources.states->ple(metadata.destination_slots), cuda);

    auto conditioning = packed_tensor(*buffers.pending_conditioning, {channels, width, batch});
    auto sample       = packed_tensor(buffers.pending_sample, {hidden, width, batch});
    for (int b = 0; b < batch; ++b) {
        const auto a = commit.rows[b].accepted;
        if (!a) continue;
        const auto slot = commit.rows[b].slots.destination;
        const auto save = [&](const Tensor& values, Tensor store) {
            const auto source      = values.slice(2, b, 1).slice(1, a - 1, 1);
            const auto destination = store.slice(1, slot, 1);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {source.data, source.bytes(), cuda.device()}, cuda);
        };
        save(sample, resources.states->sample_hidden());
        save(conditioning, resources.states->conditioning());
        auto ids = packed_tensor(*buffers.sampled_ids, {width, batch})
                       .slice(1, b, 1)
                       .slice(0, 0, a)
                       .view({static_cast<std::int32_t>(a)});
        auto counts =
            device_tensor(*buffers.token_counts).slice(1, rows[b].member.residency.lane.value, 1);
        ops::increment_token_counts(ids, counts, cuda);
    }

    // The first draft used the committed Main conditioning and anchor. Its tail is already real;
    // only the accepted later positions need rebuilding with target conditioning.
    auto tail = mtp_working_tail(0, batch);
    if (rebuild_history) {
        const auto k = width - 1;
        auto aligned = packed_tensor(frame.alignment_conditioning, {channels, k, batch});
        CUDA_CHECK(cudaMemsetAsync(aligned.data, 0, aligned.bytes(), cuda.stream()));
        std::array<MtpInputRow, kMaximumConcurrency> inputs;
        for (int b = 0; b < batch; ++b) {
            const auto a     = commit.rows[b].accepted;
            const auto count = a ? a - 1 : 0;
            inputs[b]        = {round.bases[b],
                                std::span<const TokenId>(round.inputs).subspan(b * width + 1, count)};
            if (!count) continue;
            const auto source      = conditioning.slice(2, b, 1).slice(1, 0, count);
            const auto destination = aligned.slice(2, b, 1).slice(1, 0, count);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {source.data, source.bytes(), cuda.device()}, cuda);
        }
        const auto call = mtp_inputs(frame.history, rows, k, {inputs.data(), rows.size()});
        auto rebuilt    = mtp_working_tail(1, batch);
        execution::mtp_history(parameters.profile->config.text, parameters.profile->local.qsa,
                               *parameters.mtp, aligned, call, tail, rebuilt,
                               resources.mtp->layer(0), {}, resources.workspace, cuda);
        tail = rebuilt;
    }
    const auto destination = resources.states->mtp_tail(metadata.destination_slots);
    const auto copy        = [&](const Tensor& source, const Tensor& output) {
        copy_bytes_async({output.data, output.bytes(), cuda.device()},
                                {source.data, source.bytes(), cuda.device()}, cuda);
    };
    for (int b = 0; b < batch; ++b) {
        if (!commit.rows[b].accepted) continue;
        const auto slot = commit.rows[b].slots.destination;
        copy(tail.raw_keys.slice(2, b, 1), destination.raw_keys.slice(2, slot, 1));
        copy(tail.first_rope.slice(1, b, 1), destination.first_rope.slice(1, slot, 1));
    }
}

} // namespace ninfer::models::qwen4_exp::detail
