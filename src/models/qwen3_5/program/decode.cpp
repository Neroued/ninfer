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
    const auto batch      = forward ? 1 : static_cast<int>(operation.shape.rows);
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

void RankProgramImpl::run_dflash_draft(const speculative::RoundRoute& route,
                                       std::uint32_t max_frontier) {
    auto& frame     = resources.frame;
    auto& draft     = *frame.dflash;
    const int batch = static_cast<int>(route.shape.rows);
    execution::DraftInputs inputs{frame.anchors.slice(0, 0, batch),
                                  frame.frontiers.slice(0, 0, batch),
                                  frame.proposal_valid_columns.slice(0, 0, batch),
                                  frame.state_destination_offsets.slice(0, 0, batch),
                                  frame.backend_table_rows.slice(0, 0, batch),
                                  frame.proposal_live_rows.slice(0, 0, batch),
                                  frame.sampling};
    auto outputs = draft.proposal(batch, static_cast<int>(route.head_columns));
    // A live draft row has the full proposal width. Only graph profiles can extend
    // into the frontier interval where no proposal fits.
    const auto width         = static_cast<std::uint32_t>(outputs.ids.ne[0]);
    const auto local_visible = std::min(capacity, max_frontier + width);
    const execution::DraftBuckets buckets{ops::select_context_bucket(std::max(1U, local_visible)),
                                          ops::select_context_bucket(std::max(1U, max_frontier))};
    execution::draft_propose(draft_view(*this), inputs, outputs, buckets);
}

Tensor RankProgramImpl::draft_tokens(const speculative::RoundRoute& route) const {
    const auto& frame = resources.frame;
    const auto batch  = static_cast<int>(route.shape.rows);
    if (frame.mtp) return frame.mtp->proposals(batch);
    return frame.dflash->proposal(batch, static_cast<int>(route.head_columns)).drafts;
}

void RankProgramImpl::prepare_verification(const speculative::RoundRoute& route) {
    auto& frame          = resources.frame;
    const auto batch     = static_cast<int>(route.shape.rows);
    const auto produced  = frame.mtp ? route.mtp_steps : route.head_columns;
    const auto proposals = draft_tokens(route).slice(0, 0, produced);
    auto ids             = frame.decode_view(route.width, batch).ids;
    ops::prepare_verify_ids(frame.anchors.slice(0, 0, batch), proposals,
                            frame.extents.slice(0, 0, batch), ids, device.execution_view());
}

void RankProgramImpl::begin_draft(ComputeOperation& operation) {
    auto& frame           = resources.frame;
    const int batch       = static_cast<int>(operation.shape.rows);
    std::uint32_t maximum = 0;
    for (std::size_t i = 0; i < operation.rows.size(); ++i) {
        if (!frame.host_ingress.extents[i]) continue;
        maximum = std::max(maximum, operation.rows[i].main_frontier);
    }
    if (!graphs) gpu_timer.begin_draft();
    if (graphs) {
        (void)graphs->start(GraphPhase::Draft, operation.decode_route(), maximum, 0,
                            device.execution_view());
    } else if (backend == SpeculativeBackend::Mtp) {
        run_mtp_draft(batch, operation.decode_route().mtp_steps, maximum);
    } else {
        run_dflash_draft(operation.decode_route(), maximum);
    }
    if (!graphs) {
        gpu_timer.end_draft();
        prepare_verification(operation.decode_route());
    }
    if (operation.read_drafts_for_masks) {
        const Tensor proposals = draft_tokens(operation.decode_route());
        if (backend == SpeculativeBackend::Mtp) {
            const auto row_bytes = std::size_t(batch) * sizeof(TokenId);
            CUDA_CHECK(cudaMemcpy2DAsync(frame.host_drafts.data, row_bytes, proposals.data,
                                         frame.mtp->drafts.nb[1], row_bytes,
                                         operation.decode_route().mtp_steps, cudaMemcpyDeviceToHost,
                                         device.stream()));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(frame.host_drafts.data, proposals.data, proposals.bytes(),
                                       cudaMemcpyDeviceToHost, device.stream()));
        }
        drafts_ready.record(device.stream());
    }
}

void RankProgramImpl::begin_target(ComputeOperation& operation) {
    if (!operation.width) return;
    auto& frame        = resources.frame;
    const bool forward = std::holds_alternative<ForwardWork>(operation.request.work.operation) ||
                         std::holds_alternative<ScoreWork>(operation.request.work.operation);
    const int batch = forward ? 1 : static_cast<int>(operation.shape.rows);
    const int width = static_cast<int>(operation.width);
    ++operation.attention.forwards;
    operation.attention.rows += forward ? 1 : operation.batch;
    operation.attention.scheduled_positions +=
        std::uint64_t(width) * (forward ? 1 : operation.batch);
    std::uint32_t active_tokens = forward ? width : 0;
    if (!forward)
        for (std::uint32_t row = 0; row < operation.batch; ++row)
            active_tokens += frame.host_ingress.valid_columns[row];
    operation.attention.valid_positions += active_tokens;
    auto input = forward ? frame.input_view(width, batch) : frame.decode_view(width, batch);
    if (!forward && graphs) {
        const auto phase = operation.verifies_drafts() ? GraphPhase::Verify : GraphPhase::Main;
        std::uint32_t frontier = 0;
        for (const auto& row : operation.rows) frontier = std::max(frontier, row.main_frontier);
        operation.graph_started = true;
        operation.projected     = true;
        operation.row_finished  = graphs->start(phase, operation.decode_route(), frontier,
                                                active_tokens, device.execution_view());
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
    inputs.active_tokens              = active_tokens;
    execution::TextStateUpdate update = execution::StepUpdate{};
    if (forward) {
        const auto& row = operation.rows[operation.forward_row];
        update = execution::SequenceUpdate{row.state_offsets.source, row.state_offsets.destination};
    } else if (operation.verifies_drafts()) {
        update.emplace<execution::ReplayRecords>(resources.replay(width).records);
    }
    std::uint32_t maximum = inputs.valid_columns.data ? 1U : operation.width;
    for (std::uint32_t row = 0; row < (forward ? 1 : operation.batch); ++row) {
        if (!frame.host_ingress.valid_columns[row]) continue;
        const auto end = static_cast<std::uint32_t>(frame.host_ingress.frontiers[row] +
                                                    frame.host_ingress.valid_columns[row]);
        maximum        = std::max(maximum, end);
    }
    const auto bucket = ops::select_context_bucket(maximum);
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
    if (!forward) gpu_timer.begin_target();
    operation.text = std::make_unique<execution::TextRun>(
        execution::TextExecutionView{parameters, resources.work, device.execution_view(),
                                     resources.decoder.text_kv, resources.state_images.linear(),
                                     resources.moe.get()},
        std::move(inputs), update, bucket, input.hidden, features ? &*features : nullptr);
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
