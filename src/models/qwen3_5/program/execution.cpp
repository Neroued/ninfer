#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/text.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::progress() {
    if (context_operation && context_ready.ready()) {
        const auto ticket    = context_operation->request.ticket;
        const auto command   = context_operation->last_command;
        const auto transfers = context_actions->observations();
        ContextResult result;
        result.transfers.assign(transfers.begin(), transfers.end());
        context_actions->finish();
        events.emplace_back(
            runtime::ResultReady<ExecutionTypes>{ticket, Result{std::move(result)}});
        events.emplace_back(runtime::DeviceRetired{ticket, command, retirement_observation()});
        context_operation.reset();
    }
    if (!compute) return;
    auto& operation = *compute;
    if (operation.cancelled && operation.stage != ComputeStage::Retiring) {
        if (operation.text) {
            operation.text->cancel();
            operation.text.reset();
        }
        if (graphs) graphs->cancel();
        if (resources.moe) resources.moe->cancel();
        gpu_timer.record_retirement();
        compute_retired.record(device.stream());
        operation.stage = ComputeStage::Retiring;
    }
    for (;;) {
        switch (operation.stage) {
        case ComputeStage::Inputs:
            if (std::holds_alternative<NormalizeWork>(operation.request.work.operation)) {
                append_draft_history(operation);
                gpu_timer.record_result();
                result_ready.record(device.stream());
                operation.stage = ComputeStage::Readback;
                break;
            }
            if (std::holds_alternative<DecodeWork>(operation.request.work.operation)) {
                if (!operation.speculative || !graphs) append_draft_history(operation);
                if (operation.speculative) begin_draft(operation);
            }
            if (std::holds_alternative<ForwardWork>(operation.request.work.operation))
                append_draft_history(operation);
            request_decision(operation);
            begin_target(operation);
            operation.stage = ComputeStage::Forward;
            break;
        case ComputeStage::Forward:
            if (!advance_target(operation)) {
                request_decision(operation);
                return;
            }
            if (std::holds_alternative<DecodeWork>(operation.request.work.operation)) {
                if (backend == SpeculativeBackend::Mtp && !operation.speculative)
                    align_mtp_history(operation);
                if (operation.fused_decision) {
                    gpu_timer.record_result();
                    result_ready.record(device.stream());
                    operation.stage = ComputeStage::Readback;
                } else {
                    if (!operation.projected) {
                        resources.work.reset();
                        project_decode(operation.batch, operation.width);
                        operation.projected = true;
                    }
                    operation.stage = ComputeStage::AwaitDecision;
                }
                break;
            }
            if (operation.width) {
                if (backend == SpeculativeBackend::Mtp) align_mtp_history(operation);
                if (resources.dflash) {
                    const int width = static_cast<int>(operation.width);
                    auto& draft     = *resources.dflash;
                    auto& frame     = resources.frame;
                    execution::draft_append_context(
                        {parameters, resources.work, device.execution_view(), draft.local,
                         draft.full ? &*draft.full : nullptr, proposal_head},
                        draft.prefill_features.slice(1, 0, width)
                            .view({draft.prefill_features.ne[0], width, 1}),
                        draft.prefill_positions.slice(0, 0, width).view({width, 1}),
                        frame.valid_columns.slice(0, 0, 1),
                        frame.state_destination_offsets.slice(0, 0, 1),
                        frame.backend_table_rows.slice(0, 0, 1),
                        {operation.width, operation.width});
                }
                const auto source = resources.frame.input_view(static_cast<int>(operation.width), 1)
                                        .hidden.slice(1, static_cast<int>(operation.width - 1), 1);
                const auto target = resources.state_images.continuation_hidden(
                    operation.rows[operation.forward_row].state_offsets.destination);
                CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, target.bytes(),
                                           cudaMemcpyDeviceToDevice, device.stream()));
            }
            if (operation.width) prefill_timer.record_stop();
            gpu_timer.record_result();
            result_ready.record(device.stream());
            operation.stage = ComputeStage::History;
            break;
        case ComputeStage::History:
            if (!result_ready.ready()) return;
            if (operation.width)
                operation.gpu.add_prefill(
                    static_cast<std::uint64_t>(prefill_timer.elapsed_ms() * 1.0e6));
            if (const auto* forward = std::get_if<ForwardWork>(&operation.request.work.operation)) {
                auto& result       = std::get<ForwardResult>(operation.result->output);
                const auto& window = forward->inputs[operation.forward_row];
                operation.forward_offset += operation.width;
                const auto lane = operation.rows[operation.forward_row].member.residency.lane.value;
                auto& residency = residencies[lane];
                if (residency.vision) {
                    operation.gpu.add_vision(static_cast<std::uint64_t>(
                        (residency.vision->elapsed_seconds() - operation.vision_before) * 1.0e9));
                    if (window.prompt_begin + operation.forward_offset >=
                        residency.vision_input->plan.uses.back().end) {
                        residency.vision.reset();
                        residency.vision_input.reset();
                    }
                }
                if (operation.forward_offset == window.tokens.size()) {
                    result.processed_tokens.push_back(operation.forward_offset);
                    operation.forward_offset = 0;
                    ++operation.forward_row;
                }
                if (operation.forward_row < operation.rows.size()) {
                    initialize_forward(operation, *forward);
                    append_draft_history(operation);
                    begin_target(operation);
                    operation.stage = ComputeStage::Forward;
                } else if (forward->sample != SamplePurpose::None) {
                    operation.stage = ComputeStage::AwaitDecision;
                } else {
                    operation.stage = ComputeStage::Readback;
                }
            } else if (std::holds_alternative<ScoreWork>(operation.request.work.operation)) {
                launch_score(operation);
            }
            break;
        case ComputeStage::AwaitDecision:
            request_decision(operation);
            if (!operation.decision_supplied) return;
            launch_decision(operation);
            break;
        case ComputeStage::Readback:
            if (!result_ready.ready()) return;
            if (const auto* score = std::get_if<ScoreWork>(&operation.request.work.operation)) {
                const auto* values =
                    static_cast<const float*>(resources.frame.host_score_logprobs.data);
                std::copy_n(values, operation.score_count,
                            operation.scores.begin() + operation.score_cursor);
                operation.score_cursor += operation.score_count;
                if (operation.score_cursor < score->targets.size()) {
                    launch_score(operation);
                    break;
                }
            }
            publish_result(operation);
            return;
        case ComputeStage::AwaitCommit:
            return;
        case ComputeStage::Commit:
            launch_commit(operation);
            if (!compute) return;
            break;
        case ComputeStage::Retiring:
            if (!compute_retired.ready()) return;
            retire_compute();
            return;
        }
    }
}

} // namespace ninfer::models::qwen3_5::detail
