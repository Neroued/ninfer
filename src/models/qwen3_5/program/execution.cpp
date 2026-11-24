#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/text.h"

#include <algorithm>

namespace ninfer::models::qwen3_5::detail {

void RankProgramImpl::begin_packed_target(ComputeOperation& operation) {
    const auto& ragged = operation.packed_prefill->ragged;
    const int segments = ragged.num_seqs(), tokens = ragged.total_tokens();
    auto& frame      = resources.frame;
    const auto input = frame.input_view(tokens, 1);
    const auto& meta = frame.packed_prefill->device;
    ++operation.attention.forwards;
    operation.attention.rows += segments;
    operation.attention.scheduled_positions += tokens;
    operation.attention.valid_positions += tokens;

    std::optional<execution::DFlashFeatureSink> features;
    if (resources.dflash) {
        const auto& draft = *resources.dflash;
        features.emplace(draft.prefill_features.slice(1, 0, tokens),
                         draft.prefill_positions.slice(0, 0, tokens), Tensor{},
                         parameters.profile->config.draft->target_layer_ids);
    }
    const execution::TextInputs inputs{
        .ids                       = input.ids,
        .cache_positions           = input.cache_positions,
        .rope_positions            = input.rope_positions,
        .valid_columns             = {},
        .kv_table_rows             = meta.main_tables.slice(0, 0, segments),
        .state_source_offsets      = meta.source_offsets.slice(0, 0, segments),
        .state_destination_offsets = meta.destination_offsets.slice(0, 0, segments),
        .input_embeddings          = {},
        .active_tokens             = operation.width,
    };
    resources.work.reset();
    prefill_timer.start();
    operation.text = std::make_unique<execution::TextRun>(
        execution::TextExecutionView{parameters, resources.work, device.execution_view(),
                                     resources.decoder.text_kv, resources.state_images.linear(),
                                     resources.moe.get()},
        inputs, execution::PackedPrefillUpdate{ragged}, target_bucket(), input.hidden,
        features ? &*features : nullptr);
    operation.row_finished = operation.text->start();
}

void RankProgramImpl::finish_packed_prefill(ComputeOperation& operation) {
    const auto& ragged  = operation.packed_prefill->ragged;
    const auto& frame   = resources.frame;
    const auto& storage = *frame.packed_prefill;
    const auto hidden   = frame.input_view(ragged.total_tokens(), 1).hidden;
    const auto* destinations =
        static_cast<const std::int64_t*>(storage.host.destination_offsets.data);
    if (backend == SpeculativeBackend::Mtp) align_packed_mtp_history(operation);
    for (int s = 0; s < ragged.num_seqs(); ++s) {
        const int begin = ragged.segment_begin(s), len = ragged.segment_length(s);
        if (resources.dflash) {
            auto& draft      = *resources.dflash;
            const auto count = static_cast<std::uint32_t>(len);
            execution::draft_append_context(
                {parameters, resources.work, device.execution_view(), draft.local,
                 draft.full ? &*draft.full : nullptr, proposal_head},
                draft.prefill_features.slice(1, begin, len)
                    .view({draft.prefill_features.ne[0], len, 1}),
                draft.prefill_positions.slice(0, begin, len).view({len, 1}),
                ragged.seq_lengths.slice(0, s, 1),
                storage.device.destination_offsets.slice(0, s, 1),
                storage.device.backend_tables.slice(0, s, 1), {count, count});
        }
        const auto source = hidden.slice(1, begin + len - 1, 1);
        const auto target = resources.state_images.continuation_hidden(destinations[s]);
        CUDA_CHECK(cudaMemcpyAsync(target.data, source.data, target.bytes(),
                                   cudaMemcpyDeviceToDevice, device.stream()));
    }
}

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
        if (operation.graph_target) {
            operation.graph_target->cancel();
            operation.graph_target = nullptr;
        }
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
            if (const auto* decode = std::get_if<DecodeWork>(&operation.request.work.operation)) {
                if (!operation.verifies_drafts() || !graphs) append_draft_history(operation);
                if (operation.verifies_drafts()) {
                    begin_draft(operation);
                    if (resources.moe && decode->narrowing) {
                        operation.stage = ComputeStage::Proposal;
                        break;
                    }
                }
            }
            if (std::holds_alternative<ForwardWork>(operation.request.work.operation))
                append_draft_history(operation);
            begin_target(operation);
            request_decision(operation);
            operation.stage = ComputeStage::Forward;
            break;
        case ComputeStage::Proposal:
            if (!drafts_ready.ready()) return;
            narrow_target(operation);
            begin_target(operation);
            request_decision(operation);
            operation.stage = ComputeStage::Forward;
            break;
        case ComputeStage::Forward:
            if (!advance_target(operation)) {
                request_decision(operation);
                return;
            }
            if (std::holds_alternative<DecodeWork>(operation.request.work.operation)) {
                if (!operation.projected) {
                    resources.work.reset();
                    project_decode(operation.shape.rows, operation.width);
                    gpu_timer.end_target();
                    operation.projected = true;
                }
                operation.stage = ComputeStage::AwaitDecision;
                break;
            }
            if (operation.width && operation.packed_prefill) {
                finish_packed_prefill(operation);
            } else if (operation.width) {
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
                if (operation.packed_prefill) {
                    operation.stage = operation.packed_prefill->sampled_count
                                          ? ComputeStage::AwaitDecision
                                          : ComputeStage::Readback;
                    break;
                }
                auto& result       = std::get<ForwardResult>(operation.result->output);
                const auto& window = forward->inputs[operation.forward_row];
                operation.forward_offset += operation.width;
                if (window.visual && resources.vision)
                    if (auto elapsed = resources.vision->take_timing())
                        operation.gpu.add_vision(*elapsed);
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
