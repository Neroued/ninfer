#include "models/qwen3_5/program/internal.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>
#include <array>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::launch_commit(ComputeOperation& operation) {
    auto& frame       = resources.frame;
    const auto& value = *operation.commit;
    const bool decode = std::holds_alternative<DecodeWork>(operation.request.work.operation);
    bool device_work  = false;
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
        if (mtp_history || draft_history) {
            *frame.host_commit = {};
            for (std::uint32_t i = 0; i < operation.batch; ++i) {
                const auto& row    = operation.rows[i];
                const int accepted = static_cast<int>(value.rows[i].accepted_tokens);
                frame.host_commit->history_columns[i]         = std::max(accepted - 1, 0);
                frame.host_commit->state_destination_slots[i] = row.state_slots.destination;
                frame.host_commit->draft_append_starts[i]     = static_cast<int>(row.main_frontier);
                frame.host_commit->draft_append_ends[i] =
                    static_cast<int>(row.main_frontier) +
                    (value.rows[i].terminal && !value.rows[i].discarded ? accepted : 0);
            }
            CUDA_CHECK(cudaMemcpyAsync(frame.commit.data, frame.host_commit, sizeof(FrameCommit),
                                       cudaMemcpyHostToDevice, device.stream()));
            device_work = true;
        }
        if (operation.speculative) {
            std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold{};
            for (std::uint32_t i = 0; i < operation.batch; ++i) {
                const auto& row = operation.rows[i];
                fold[i]         = {row.state_slots.source, row.state_slots.destination,
                                   static_cast<int>(value.rows[i].accepted_tokens)};
            }
            resources.replay_fold->execute(std::span(fold).first(operation.batch),
                                           device.execution_view());
            device_work = true;
            if (mtp_history) {
                std::uint32_t columns = 0;
                for (const auto& row : value.rows)
                    if (!row.discarded && row.accepted_tokens)
                        columns = std::max(columns, row.accepted_tokens - 1);
                if (graphs)
                    graphs->rebuild_mtp_history(operation.batch, columns, device.execution_view());
                else
                    rebuild_mtp_history(operation.batch, columns);
            }
        }
        const auto hidden =
            frame.input_view(static_cast<int>(operation.width), static_cast<int>(operation.batch))
                .hidden;
        for (std::uint32_t i = 0; i < operation.batch; ++i) {
            const auto accepted = value.rows[i].accepted_tokens;
            if (!accepted) continue;
            if (backend == SpeculativeBackend::Mtp ||
                accepted < static_cast<std::uint32_t>(frame.host_egress->counts[i])) {
                Tensor source = hidden.slice(2, static_cast<int>(i), 1)
                                    .slice(1, static_cast<int>(accepted - 1), 1);
                Tensor destination = resources.state_images.continuation_hidden_slot(
                    operation.rows[i].state_slots.destination);
                CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, destination.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream()));
                device_work = true;
            }
            if (backend == SpeculativeBackend::DFlash2 && operation.speculative &&
                frame.host_ingress->sampling[i].token_counts) {
                Tensor ids = frame.output_tokens.slice(1, static_cast<int>(i), 1)
                                 .slice(0, 0, static_cast<int>(accepted))
                                 .view({static_cast<int>(accepted)});
                Tensor counts =
                    resources.token_counts
                        .slice(1, static_cast<int>(operation.rows[i].member.residency.lane.value),
                               1)
                        .view({execution::dimension(parameters.profile->public_tokens)});
                ops::increment_token_counts(ids, counts, device.execution_view());
                device_work = true;
            }
        }
        if (draft_history) commit_draft_history(operation);
    }
    if (const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation);
        forward && forward->purpose == ForwardPurpose::Control) {
        for (std::size_t i = 0; i < forward->committed_tokens.size(); ++i) {
            const auto& tokens = forward->committed_tokens[i];
            if (tokens.empty() ||
                !residencies[operation.rows[i].member.residency.lane.value].penalties)
                continue;
            auto host = frame.host_control_tokens.slice(1, static_cast<int>(i), 1)
                            .slice(0, 0, static_cast<int>(tokens.size()));
            std::copy(tokens.begin(), tokens.end(), static_cast<TokenId*>(host.data));
            auto ids = frame.input_ids.slice(0, 0, static_cast<int>(tokens.size()));
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
    compute_retired.record(device.stream());
    operation.stage = ComputeStage::Retiring;
}

void RankProgramImpl::retire_compute() {
    auto ticket  = compute->request.ticket;
    auto command = compute->last_command;
    if (compute_actions->occupied()) compute_actions->finish();
    if (compute->commit) {
        for (std::size_t i = 0; i < compute->commit->rows.size(); ++i) {
            if (!compute->commit->rows[i].terminal && !compute->commit->rows[i].discarded) continue;
            auto& residency = residencies[compute->rows[i].member.residency.lane.value];
            residency.vision.reset();
            residency.vision_input.reset();
        }
    }
    compute.reset();
    resources.work.reset();
    events.emplace_back(runtime::DeviceRetired{ticket, command});
}

} // namespace ninfer::models::qwen3_5::detail
