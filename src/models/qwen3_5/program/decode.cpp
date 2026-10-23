#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/draft.h"
#include "models/qwen3_5/execution/text.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {

execution::DraftExecutionView draft_view(RankProgramImpl& program) {
    auto& state = *program.resources.dflash;
    return {program.parameters,
            program.resources.work,
            program.device.execution_view(),
            state.local,
            state.full ? &*state.full : nullptr,
            program.proposal_head};
}

Tensor dense_prefix(const Tensor& tensor, std::initializer_list<std::int32_t> shape) {
    return Tensor({tensor.data, tensor.bytes(), tensor.device}, tensor.dtype, shape);
}

} // namespace

void RankProgramImpl::append_draft_history(ComputeOperation& operation) {
    if (!resources.dflash) return;
    const bool forward    = std::holds_alternative<ForwardWork>(operation.request.work.operation);
    const auto batch      = forward ? 1 : static_cast<int>(operation.batch);
    std::uint32_t minimum = std::numeric_limits<std::uint32_t>::max(), maximum = 0;
    const auto rows = forward ? std::span(operation.rows).subspan(operation.forward_row, 1)
                              : std::span(operation.rows);
    for (const auto& row : rows) {
        const auto count = row.main_frontier - row.backend_frontier;
        minimum          = std::min(minimum, count);
        maximum          = std::max(maximum, count);
    }
    if (!maximum) return;
    append_draft_history(batch, {minimum, maximum});
}

void RankProgramImpl::append_draft_history(std::uint32_t batch_size,
                                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    auto& frame      = resources.frame;
    auto& state      = *resources.dflash;
    auto& draft      = *frame.dflash;
    const auto batch = static_cast<std::int32_t>(batch_size);
    const auto width = static_cast<std::int32_t>(draft_window + 1);
    auto scope       = resources.work.scope();
    auto features = resources.work.alloc(DType::BF16, {state.pending_features.ne[0], width, batch});
    auto positions = dense_prefix(draft.append_positions, {width, batch});
    auto counts    = draft.append_counts.slice(0, 0, batch);
    ops::prepare_ragged_prefix(state.pending_features, frame.residency_slots.slice(0, 0, batch),
                               frame.backend_frontiers.slice(0, 0, batch),
                               frame.frontiers.slice(0, 0, batch), features, positions, counts,
                               device.execution_view());
    execution::draft_append_context(draft_view(*this), features, positions, counts,
                                    frame.state_destination_offsets.slice(0, 0, batch),
                                    frame.backend_table_rows.slice(0, 0, batch), envelope);
}

void RankProgramImpl::run_dflash_draft(std::uint32_t batch_value, std::uint32_t minimum,
                                       std::uint32_t maximum) {
    auto& frame      = resources.frame;
    auto& draft      = *frame.dflash;
    const int batch  = static_cast<int>(batch_value);
    const int drafts = static_cast<int>(draft_window), width = drafts + 1;
    execution::DraftInputs inputs{frame.anchors.slice(0, 0, batch),
                                  frame.frontiers.slice(0, 0, batch),
                                  (backend == SpeculativeBackend::DFlash2
                                       ? frame.proposal_valid_columns
                                       : frame.valid_columns)
                                      .slice(0, 0, batch),
                                  frame.state_destination_offsets.slice(0, 0, batch),
                                  frame.backend_table_rows.slice(0, 0, batch),
                                  frame.sampling};
    execution::DraftOutputs outputs{
        dense_prefix(draft.proposal_ids, {width, batch}),
        dense_prefix(draft.proposal_positions, {width, batch}),
        dense_prefix(draft.drafts, {drafts, batch}),
        draft.candidate_ids.data
            ? dense_prefix(draft.candidate_ids, {draft.candidate_ids.ne[0], drafts, batch})
            : Tensor{},
        draft.proposal_q.data
            ? dense_prefix(draft.proposal_q, {draft.proposal_q.ne[0], drafts, batch})
            : Tensor{}};
    execution::DraftEnvelopes envelopes{{minimum, maximum}, {minimum, maximum}};
    execution::draft_propose(draft_view(*this), inputs, outputs, envelopes);
}

void RankProgramImpl::begin_draft(ComputeOperation& operation) {
    auto& frame           = resources.frame;
    const int batch       = static_cast<int>(operation.batch);
    const int drafts      = static_cast<int>(draft_window);
    const int width       = drafts + 1;
    std::uint32_t minimum = capacity, maximum = 0;
    for (const auto& row : operation.rows) {
        minimum = std::min(minimum, row.main_frontier);
        maximum = std::max(maximum, row.main_frontier);
    }
    if (graphs) {
        if (backend == SpeculativeBackend::Mtp) prepare_mtp_draft(operation);
        (void)graphs->start(GraphPhase::Draft, operation.batch, maximum, device.execution_view());
    } else if (backend == SpeculativeBackend::Mtp) {
        begin_mtp_draft(operation);
    } else {
        run_dflash_draft(operation.batch, minimum, maximum);
    }
    Tensor proposals =
        dense_prefix(backend == SpeculativeBackend::Mtp ? frame.mtp->drafts : frame.dflash->drafts,
                     {drafts, batch});
    auto input = frame.input_view(width, batch);
    ops::prepare_verify_ids(frame.anchors.slice(0, 0, batch), proposals,
                            frame.extents.slice(0, 0, batch), input.ids, device.execution_view());
    const bool constrained = std::any_of(operation.sampling.begin(), operation.sampling.end(),
                                         [](const auto& row) { return row.constrained; });
    if (constrained) {
        CUDA_CHECK(cudaMemcpyAsync(frame.host_drafts.data, proposals.data, proposals.bytes(),
                                   cudaMemcpyDeviceToHost, device.stream()));
        drafts_ready.record(device.stream());
    }
}

void RankProgramImpl::begin_target(ComputeOperation& operation) {
    if (!operation.width) return;
    auto& frame        = resources.frame;
    const bool forward = std::holds_alternative<ForwardWork>(operation.request.work.operation) ||
                         std::holds_alternative<ScoreWork>(operation.request.work.operation);
    const int batch = forward ? 1 : static_cast<int>(operation.batch);
    const int width = static_cast<int>(operation.width);
    auto input      = frame.input_view(width, batch);
    if (!forward && graphs) {
        operation.fused_decision =
            !operation.speculative &&
            std::none_of(operation.sampling.begin(), operation.sampling.end(),
                         [](const auto& row) { return row.constrained; });
        const auto phase       = operation.speculative      ? GraphPhase::Verify
                                 : operation.fused_decision ? GraphPhase::Ordinary
                                                            : GraphPhase::Main;
        std::uint32_t frontier = 0;
        for (const auto& row : operation.rows) frontier = std::max(frontier, row.main_frontier);
        operation.graph_started = true;
        operation.projected     = true;
        operation.row_finished =
            graphs->start(phase, operation.batch, frontier, device.execution_view());
        return;
    }
    execution::TextInputs inputs{input.ids,
                                 input.cache_positions,
                                 input.rope_positions,
                                 frame.valid_columns.slice(0, 0, batch),
                                 frame.main_table_rows.slice(0, 0, batch),
                                 frame.state_source_offsets.slice(0, 0, batch),
                                 frame.state_destination_offsets.slice(0, 0, batch),
                                 operation.input_embeddings};
    execution::TextStateUpdate update = execution::StepUpdate{};
    if (forward) {
        const auto& row = operation.rows[operation.forward_row];
        update = execution::SequenceUpdate{row.state_offsets.source, row.state_offsets.destination};
    } else if (operation.speculative) {
        update.emplace<execution::ReplayRecords>(*resources.replay_records);
    }
    std::uint32_t minimum = capacity, maximum = 0;
    for (int row = 0; row < batch; ++row) {
        const auto end = static_cast<std::uint32_t>(frame.host_ingress->frontiers[row] +
                                                    frame.host_ingress->valid_columns[row]);
        minimum        = std::min(minimum, end);
        maximum        = std::max(maximum, end);
    }
    std::optional<execution::DFlashFeatureSink> features;
    if (resources.dflash) {
        auto& state = *resources.dflash;
        features.emplace();
        features->layers = parameters.profile->config.draft->target_layer_ids;
        if (forward) {
            features->features  = state.prefill_features.slice(1, 0, width);
            features->positions = state.prefill_positions.slice(0, 0, width);
        } else {
            features->features        = state.pending_features.slice(1, 0, width);
            features->residency_slots = frame.residency_slots.slice(0, 0, batch);
        }
    }
    resources.work.reset();
    if (forward) prefill_timer.start();
    operation.text = std::make_unique<execution::TextRun>(
        execution::TextExecutionView{parameters, resources.work, device.execution_view(),
                                     resources.decoder.text_kv, resources.state_images.linear(),
                                     resources.moe.get()},
        std::move(inputs), update, ops::CausalAttentionExecutionEnvelope{minimum, maximum},
        input.hidden, features ? &*features : nullptr);
    operation.row_finished = operation.text->start();
}

bool RankProgramImpl::advance_target(ComputeOperation& operation) {
    if (!operation.width) return true;
    if (!operation.row_finished)
        operation.row_finished = operation.graph_started ? graphs->poll() : operation.text->poll();
    if (!operation.row_finished) return false;
    workspace_peak = std::max(workspace_peak, resources.work.peak_used());
    operation.text.reset();
    return true;
}

} // namespace ninfer::models::qwen3_5::detail
