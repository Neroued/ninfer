#include "models/qwen3_5/program/internal.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/indexing.h"

#include <algorithm>
#include <array>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::launch_commit(ComputeOperation& operation) {
    auto& frame          = resources.frame;
    const auto& value    = *operation.commit;
    const bool decode    = std::holds_alternative<DecodeWork>(operation.request.work.operation);
    const auto* forward  = std::get_if<ForwardWork>(&operation.request.work.operation);
    const bool generated = decode || (forward && forward->sample != SamplePurpose::None);
    const bool mtp       = decode && operation.speculative && backend == SpeculativeBackend::Mtp;
    bool count_tokens    = false;
    *frame.host_commit   = {};
    for (std::uint32_t i = 0; generated && i < operation.batch; ++i) {
        const auto& row     = operation.rows[i];
        const auto accepted = value.rows[i].discarded ? 0 : value.rows[i].accepted_tokens;
        frame.host_commit->state_destination_offsets[i] = row.state_offsets.destination;
        frame.host_commit->last_columns[i]              = static_cast<int>(accepted) - 1;
        frame.host_commit->accepted_columns[i]          = accepted;
        frame.host_commit->draft_append_starts[i]       = static_cast<int>(row.main_frontier);
        frame.host_commit->draft_append_ends[i] =
            static_cast<int>(row.main_frontier) + (value.rows[i].terminal ? accepted : 0);
        if (generated && frame.host_ingress->sampling[i].token_counts) {
            frame.host_commit->count_columns[i] = accepted;
            count_tokens |= accepted != 0;
        }
    }
    bool device_work = decode || count_tokens;
    if (device_work)
        CUDA_CHECK(cudaMemcpyAsync(frame.commit.data, frame.host_commit, sizeof(FrameCommit),
                                   cudaMemcpyHostToDevice, device.stream()));
    if (decode) {
        const bool mtp_history =
            backend == SpeculativeBackend::Mtp && operation.speculative &&
            std::any_of(value.rows.begin(), value.rows.end(),
                        [](const auto& row) { return !row.discarded && row.accepted_tokens > 1; });
        const bool draft_history =
            resources.dflash &&
            std::any_of(value.rows.begin(), value.rows.end(), [](const auto& row) {
                return row.terminal && !row.discarded && row.accepted_tokens;
            });
        if (mtp) {
            if (graphs)
                graphs->commit_mtp(operation.batch, mtp_history, count_tokens,
                                   device.execution_view());
            else
                execute_mtp_commit(operation.batch, mtp_history, count_tokens);
        } else {
            if (operation.speculative)
                resources.replay_fold->execute(
                    frame.state_source_offsets.slice(0, 0, operation.batch),
                    frame.commit_destination_offsets.slice(0, 0, operation.batch),
                    frame.accepted_columns.slice(0, 0, operation.batch), device.execution_view());
            const auto hidden = frame.decode_view(operation.width, operation.batch).hidden;
            ops::scatter_batch_columns(
                hidden, frame.last_columns.slice(0, 0, operation.batch),
                resources.state_images.continuation_hidden(
                    frame.commit_destination_offsets.slice(0, 0, operation.batch)),
                device.execution_view());
        }
        if (draft_history) commit_draft_history(operation);
    }
    if (count_tokens && !mtp) {
        const int width = operation.speculative ? static_cast<int>(draft_window + 1) : 1;
        Tensor tokens({frame.output_tokens.data, frame.output_tokens.bytes(), device.device()},
                      DType::I32, {width, static_cast<int>(operation.batch)});
        ops::increment_token_counts_batch(tokens, frame.count_columns.slice(0, 0, operation.batch),
                                          frame.residency_slots.slice(0, 0, operation.batch),
                                          resources.token_counts, device.execution_view());
    }
    if (const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation);
        forward && forward->purpose == ForwardPurpose::Control) {
        for (std::size_t i = 0; i < forward->control_last_tokens.size(); ++i) {
            const ControlTokens tokens{forward->inputs[i].tokens, forward->control_last_tokens[i]};
            if (!tokens.size() ||
                !residencies[operation.rows[i].member.residency.lane.value].penalties)
                continue;
            auto host = frame.host_control_tokens.slice(1, static_cast<int>(i), 1)
                            .slice(0, 0, static_cast<int>(tokens.size()));
            tokens.copy({static_cast<TokenId*>(host.data), tokens.size()});
            auto ids = frame.control_ids.slice(0, 0, static_cast<int>(tokens.size()));
            CUDA_CHECK(cudaMemcpyAsync(ids.data, host.data, ids.bytes(), cudaMemcpyHostToDevice,
                                       device.stream()));
            auto counts =
                resources.token_counts
                    .slice(1, static_cast<int>(operation.rows[i].member.residency.lane.value), 1)
                    .view({execution::dimension(parameters.profile->public_tokens)});
            ops::increment_token_counts(ids, counts, device.execution_view());
            device_work = true;
        }
    }
    if (compute_actions->occupied()) compute_actions->finish();
    if (!value.context.operations.empty()) {
        compute_actions->submit(value.context);
        device_work = true;
    }
    if (!device_work) {
        retire_compute();
        return;
    }
    gpu_timer.record_retirement();
    compute_retired.record(device.stream());
    operation.stage = ComputeStage::Retiring;
}

void RankProgramImpl::execute_mtp_commit(std::uint32_t batch, bool repair, bool counts,
                                         DecodeGraphSection* repair_section,
                                         DecodeGraphSection* counts_section) {
    auto& frame     = resources.frame;
    const auto cuda = device.execution_view();
    resources.replay_fold->execute(frame.state_source_offsets.slice(0, 0, batch),
                                   frame.commit_destination_offsets.slice(0, 0, batch),
                                   frame.accepted_columns.slice(0, 0, batch), cuda);
    const auto history = [&] { rebuild_mtp_history(batch); };
    if (repair_section)
        repair_section->capture(cuda.stream(), history);
    else if (repair)
        history();
    const auto hidden = frame.decode_view(draft_window + 1, batch).hidden;
    ops::scatter_batch_columns(hidden, frame.last_columns.slice(0, 0, batch),
                               resources.state_images.continuation_hidden(
                                   frame.commit_destination_offsets.slice(0, 0, batch)),
                               cuda);
    const auto count = [&] {
        const int width = draft_window + 1;
        Tensor tokens({frame.output_tokens.data, frame.output_tokens.bytes(), cuda.device()},
                      DType::I32, {width, static_cast<int>(batch)});
        ops::increment_token_counts_batch(tokens, frame.count_columns.slice(0, 0, batch),
                                          frame.residency_slots.slice(0, 0, batch),
                                          resources.token_counts, cuda);
    };
    if (counts_section)
        counts_section->capture(cuda.stream(), count);
    else if (counts)
        count();
}

void RankProgramImpl::retire_compute() {
    auto ticket  = compute->request.ticket;
    auto command = compute->last_command;
    if (compute_actions->occupied()) compute_actions->finish();
    if (resources.vision) {
        const auto* active = resources.vision->current();
        if (active) {
            const bool reads_window =
                std::any_of(compute->rows.begin(), compute->rows.end(), [&](const auto& row) {
                    return row.member.residency == active->owner.residency;
                });
            bool release = compute->cancelled && reads_window;
            if (const auto* forward = std::get_if<ForwardWork>(&compute->request.work.operation))
                for (const auto& input : forward->inputs)
                    if (input.visual && input.visual->window.id == active->id)
                        release |=
                            input.prompt_begin + input.tokens.size() >= active->plan->consume_end;
            if (compute->commit)
                for (std::size_t i = 0; i < compute->commit->rows.size(); ++i)
                    if (compute->rows[i].member.residency == active->owner.residency)
                        release |=
                            compute->commit->rows[i].terminal || compute->commit->rows[i].discarded;
            if (release) resources.vision->release(active->id);
        }
    }
    auto observation = retirement_observation();
    gpu_timer.observe_retirement(compute->gpu);
    observation.gpu  = compute->gpu;
    observation.path = compute->graph_started
                           ? (resources.moe ? ExecutionPath::Mixed : ExecutionPath::Graph)
                           : ExecutionPath::Eager;
    compute.reset();
    resources.work.reset();
    events.emplace_back(runtime::DeviceRetired{ticket, command, observation});
}

} // namespace ninfer::models::qwen3_5::detail
