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

execution::DraftInputs draft_inputs(const ExecutionFrame& frame, int batch) {
    return {frame.anchors.slice(0, 0, batch),
            frame.frontiers.slice(0, 0, batch),
            frame.proposal_valid_columns.slice(0, 0, batch),
            frame.state_destination_offsets.slice(0, 0, batch),
            frame.backend_table_rows.slice(0, 0, batch),
            frame.proposal_live_rows.slice(0, 0, batch),
            frame.sampling};
}

} // namespace

void RankProgramImpl::append_draft_history(ComputeOperation& operation) {
    if (!resources.dflash) return;
    const bool forward = std::holds_alternative<ForwardWork>(operation.request.work.operation);
    const bool sequential_forward = forward && !operation.packed_prefill;
    const auto batch              = sequential_forward ? 1 : static_cast<int>(operation.shape.rows);
    std::uint32_t minimum = std::numeric_limits<std::uint32_t>::max(), maximum = 0;
    const auto rows = sequential_forward
                          ? std::span(operation.rows).subspan(operation.forward_row, 1)
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

void RankProgramImpl::bind_dflash_draft(const speculative::RoundRoute& route,
                                        const execution::DraftReadoutSubmit& submit) {
    const auto& frame  = resources.frame;
    const auto batch   = static_cast<int>(route.shape.rows);
    const auto outputs = frame.dflash->proposal(batch, static_cast<int>(route.head_columns));
    execution::bind_draft_readouts(draft_view(*this), draft_inputs(frame, batch), outputs, submit);
}

void RankProgramImpl::run_dflash_draft(const speculative::RoundRoute& route,
                                       std::span<const ops::ContextBucket> buckets,
                                       const execution::DraftReadoutSubmit& submit) {
    const auto& frame   = resources.frame;
    const auto batch    = static_cast<int>(route.shape.rows);
    auto outputs        = frame.dflash->proposal(batch, static_cast<int>(route.head_columns));
    outputs.confidences = frame.draft_confidences.slice(0, 0, batch * route.head_columns)
                              .view({static_cast<int>(route.head_columns), batch});
    execution::draft_propose(draft_view(*this), draft_inputs(frame, batch), outputs,
                             {buckets.front(), buckets.back()}, submit);
}

Tensor RankProgramImpl::draft_tokens(const speculative::RoundRoute& route) const {
    const auto& frame = resources.frame;
    const auto batch  = static_cast<int>(route.shape.rows);
    if (frame.mtp) return frame.mtp->proposals(batch);
    return frame.dflash->proposal(batch, static_cast<int>(route.head_columns)).drafts;
}

void RankProgramImpl::prepare_verification(const speculative::RoundRoute& target,
                                           const speculative::RoundRoute& proposal) {
    auto& frame          = resources.frame;
    const auto batch     = static_cast<int>(target.shape.rows);
    const auto produced  = proposal.mtp_steps + proposal.head_columns;
    const auto proposals = draft_tokens(proposal).slice(0, 0, std::min(produced, target.width - 1));
    auto ids             = frame.decode_view(target.width, batch).ids;
    ops::prepare_verify_ids(frame.anchors.slice(0, 0, batch), proposals,
                            frame.extents.slice(0, 0, batch), ids, device.execution_view());
}

ProposalBuckets RankProgramImpl::proposal_buckets(const speculative::RoundRoute& route) const {
    const auto& ingress = resources.frame.host_ingress;
    ProposalBuckets out;
    if (route.mtp_steps) {
        out.count = route.mtp_steps;
        for (std::uint32_t step = 0; step < route.mtp_steps; ++step) {
            std::uint32_t maximum = 1;
            for (std::uint32_t row = 0; row < route.shape.rows; ++row) {
                if (ingress.extents[row] > static_cast<int>(step))
                    maximum = std::max(maximum,
                                       static_cast<std::uint32_t>(ingress.frontiers[row]) + step);
            }
            out.values[step] = ops::select_context_bucket(std::min(capacity, maximum));
        }
    } else if (route.head_columns) {
        std::uint32_t frontier = 0;
        for (std::uint32_t row = 0; row < route.shape.rows; ++row)
            if (ingress.extents[row])
                frontier = std::max(frontier, static_cast<std::uint32_t>(ingress.frontiers[row]));
        const auto local_visible = std::min(capacity, frontier + draft_window + 1);
        out.values[0]            = ops::select_context_bucket(std::max(1U, local_visible));
        out.count                = 1;
        if (backend == SpeculativeBackend::DFlash) {
            out.values[1] = ops::select_context_bucket(std::max(1U, frontier));
            out.count     = 2;
        }
    }
    return out;
}

ops::ContextBucket RankProgramImpl::target_bucket() const {
    const auto& ingress   = resources.frame.host_ingress;
    std::uint32_t maximum = 1;
    for (std::size_t row = 0; row < ingress.valid_columns.size(); ++row) {
        if (ingress.valid_columns[row] <= 0) continue;
        const auto end =
            static_cast<std::uint32_t>(ingress.frontiers[row] + ingress.valid_columns[row]);
        maximum = std::max(maximum, end);
    }
    return ops::select_context_bucket(maximum);
}

void RankProgramImpl::readback_drafts(const speculative::RoundRoute& route) {
    const auto& frame    = resources.frame;
    const auto proposals = draft_tokens(route);
    if (backend == SpeculativeBackend::Mtp) {
        const auto row_bytes = std::size_t(route.shape.rows) * sizeof(TokenId);
        CUDA_CHECK(cudaMemcpy2DAsync(frame.host_drafts.data, row_bytes, proposals.data,
                                     frame.mtp->drafts.nb[1], row_bytes, route.mtp_steps,
                                     cudaMemcpyDeviceToHost, device.stream()));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(frame.host_drafts.data, proposals.data, proposals.bytes(),
                                   cudaMemcpyDeviceToHost, device.stream()));
    }
}

void RankProgramImpl::begin_draft(ComputeOperation& operation) {
    const auto& route = operation.proposal_route();
    if (graphs && graphs->topology() != ProposalTopology::SplitProposal) return;
    const auto buckets = proposal_buckets(route);
    const bool narrow =
        resources.moe && std::get<DecodeWork>(operation.request.work.operation).narrowing;
    if (graphs) {
        graphs->launch_proposal(
            route, {.buckets = buckets.span(), .readback = operation.read_drafts_for_masks});
    } else {
        gpu_timer.begin_draft();
        if (backend == SpeculativeBackend::Mtp) {
            gather_mtp_conditioning(route.shape.rows);
            for (std::uint32_t step = 0; step < route.mtp_steps; ++step)
                run_mtp_draft_step(route.shape.rows, step, buckets.values[step]);
        } else {
            run_dflash_draft(route, buckets.span());
        }
        gpu_timer.end_draft();
        if (operation.read_drafts_for_masks) readback_drafts(route);
    }
    if (narrow) {
        const auto& frame = resources.frame;
        const auto count  = std::size_t(route.shape.rows) * (route.mtp_steps + route.head_columns);
        CUDA_CHECK(cudaMemcpyAsync(frame.host_egress.draft_confidences.data(),
                                   frame.draft_confidences.data, count * sizeof(float),
                                   cudaMemcpyDeviceToHost, device.stream()));
    }
    if (narrow || (!graphs && operation.read_drafts_for_masks))
        drafts_ready.record(device.stream());
}

void RankProgramImpl::narrow_target(ComputeOperation& operation) {
    const auto& work     = std::get<DecodeWork>(operation.request.work.operation);
    const auto& input    = *work.narrowing;
    const auto& proposal = operation.proposal_route();
    const auto stride    = proposal.mtp_steps + proposal.head_columns;
    std::vector<speculative::AdaptiveRowInput> rows;
    rows.reserve(input.rows.size());
    for (const auto& row : input.rows) rows.push_back({row.member, row.frontier, &row.state});
    auto confidences = std::span<const float>(resources.frame.host_egress.draft_confidences);
    std::vector<float> row_major;
    if (proposal.mtp_steps) {
        row_major.resize(rows.size() * stride);
        for (std::size_t row = 0; row < rows.size(); ++row)
            for (std::uint32_t step = 0; step < stride; ++step)
                row_major[row * stride + step] = confidences[step * proposal.shape.rows + row];
        confidences = row_major;
    }
    operation.narrowed = speculative::narrow_after_proposal(
        backend, draft_window, 1, work.geometry->path, decode_routes.get(), input.cost, *work.plan,
        rows, confidences, stride, input.max_context);
    operation.width = operation.decode_route().width;
    prepare_target_ingress(operation, operation.narrowed->verify_drafts);
    if (!graphs || resources.dflash || operation.width > 1) upload_ingress();
}

void RankProgramImpl::begin_target(ComputeOperation& operation) {
    if (!operation.width) return;
    if (operation.packed_prefill) {
        begin_packed_target(operation);
        return;
    }
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
    auto input        = forward ? frame.input_view(width, batch) : frame.decode_view(width, batch);
    const auto bucket = target_bucket();
    if (!forward && graphs) {
        const auto& route = operation.decode_route();
        TargetReplay replay{.bucket        = bucket,
                            .head          = true,
                            .active_tokens = static_cast<std::int32_t>(active_tokens)};
        ProposalBuckets buckets;
        if (operation.verifies_drafts() && graphs->topology() != ProposalTopology::SplitProposal) {
            buckets         = proposal_buckets(route);
            replay.proposal = {.buckets  = buckets.span(),
                               .readback = operation.read_drafts_for_masks};
        }
        if (resources.moe) operation.graph_target = &graphs->target(route);
        operation.graph_started = true;
        operation.projected     = true;
        operation.row_finished  = graphs->start_target(route, replay) == TargetProgress::Complete;
        if (operation.row_finished) operation.graph_target = nullptr;
        return;
    }
    if (!forward && operation.verifies_drafts())
        prepare_verification(operation.decode_route(), operation.proposal_route());
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
    if (!operation.row_finished) {
        operation.row_finished = operation.graph_target
                                     ? operation.graph_target->poll() == TargetProgress::Complete
                                     : operation.text->poll();
    }
    if (!operation.row_finished) return false;
    workspace_peak         = std::max(workspace_peak, resources.work.peak_used());
    operation.graph_target = nullptr;
    operation.text.reset();
    return true;
}

} // namespace ninfer::models::qwen3_5::detail
