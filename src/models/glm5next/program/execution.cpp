#include "models/glm5next/program/internal.h"

#include "core/nvtx.h"
#include "core/transfer.h"
#include "models/glm5next/state/continuation.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
void RankProgramImpl::execute_proposal(const execution::BatchView& main,
                                       std::span<const std::uint32_t> draft_bounds,
                                       int head_columns, bool read_drafts) {
    const int rows = main.rows, local_rows = main.attention.rows;
    begin_proposal(main, false);
    const auto drafts =
        draft_proposals(rows, parameters.mtp ? dimension(shape.draft_tokens) : head_columns);
    if (parameters.mtp) {
        nvtx::ScopedRange range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp);
        for (std::uint32_t step = 0; step < draft_bounds.size(); ++step) {
            const auto call =
                batch(frame.mtp->draft_steps[step], 1, rows, local_rows, draft_bounds[step]);
            mtp_forward(step, call);
        }
    } else {
        const auto bucket = ops::select_context_bucket(std::max(1U, draft_bounds.front()));
        dflash_forward(rows, bucket, head_columns);
    }
    timer.end_draft();
    if (read_drafts) {
        readback_drafts(drafts);
        drafts_done.record(cuda.stream());
    }
    prepare_verify_inputs(main, drafts);
}

void RankProgramImpl::begin_proposal(const execution::BatchView& main, bool capture) {
    upload_region(frame.ingress, frame.ingress.bytes);
    if (parameters.mtp) gather_decode_state(main);
    timer.begin_draft(capture);
}

void RankProgramImpl::readback_drafts(const Tensor& drafts) {
    const auto& region = frame.speculative->drafts;
    const auto bytes   = drafts.numel() * sizeof(TokenId);
    copy_bytes_async(output(region).subspan(0, bytes),
                     region.region.bind(resources.fixed()).subspan(0, bytes), cuda);
}

void RankProgramImpl::prepare_verify_inputs(const execution::BatchView& main,
                                            const Tensor& drafts) {
    auto ids = main.ids;
    ops::prepare_verify_ids(row(frame.rows.anchors, main.rows), drafts,
                            row(frame.speculative->extents, main.rows), ids, cuda);
}

void RankProgramImpl::execute_target(const execution::BatchView& main, bool head,
                                     bool forced_control) {
    auto hidden = target_hidden(main);
    begin_target(main, false);
    target_forward(main, hidden, forced_control);
    if (head) project_head(hidden.view({hidden.ne[0], main.tokens * main.rows}));
    timer.end_target();
}

void RankProgramImpl::begin_target(const execution::BatchView& main, bool capture) {
    if (main.tokens == 1) {
        upload_region(frame.ingress, frame.ingress.bytes);
        if (parameters.mtp) gather_decode_state(main);
    }
    timer.begin_target(capture);
}

Tensor RankProgramImpl::target_hidden(const execution::BatchView& main) {
    const auto hidden = dimension(parameters.profile->config.text.hidden);
    return resources.packed(frame.pending_hidden, {hidden, main.tokens, main.rows});
}

execution::StateUpdate RankProgramImpl::target_update(int width) {
    if (width > 1) return execution::RecordWindow{resources.replay(width).records};
    return execution::StepUpdate{};
}

void RankProgramImpl::target_forward(const execution::BatchView& main, Tensor hidden,
                                     bool forced_control, const execution::MlaSubmit& submit) {
    const auto w      = main.tokens;
    const auto update = target_update(w);
    if (w > 1) {
        nvtx::ScopedRange range(parameters.mtp ? nvtx::Name::DecodeMtpTarget
                                               : nvtx::Name::DecodeDFlashTarget,
                                parameters.mtp ? nvtx::Category::Mtp : nvtx::Category::DFlash);
        main_forward(main, update, hidden, {}, submit);
    } else {
        main_forward(main, update, hidden, {}, submit);
        if (forced_control || !frame.speculative) {
            if (parameters.mtp) align_history(main, hidden, 1);
            if (parameters.draft) append_dflash_context(main, main.valid_columns, {0, 1});
            ops::scatter(hidden.view({hidden.ne[0], main.rows}),
                         resources.states.hidden(main.destination_offsets), main.live_rows, cuda);
        }
    }
}

void RankProgramImpl::forward() {
    auto& f             = *compute;
    const auto& work    = f.request.work;
    auto rows           = work_rows(work);
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    if (forward && forward->kind == ForwardKind::Prefill && !forward->visual) {
        forward_packed_prefill(*forward);
        return;
    }
    const bool control =
        forward && forward->kind == ForwardKind::Control && rows.front().ids.size() > 1;
    f.sequence = std::holds_alternative<ScoreWork>(work.operation) || control ||
                 (forward &&
                  (forward->kind == ForwardKind::Prefill || forward->kind == ForwardKind::Replay));
    if (control) {
        const auto& original = rows[f.control_row];
        auto end =
            std::min<std::uint32_t>(original.ids.size(), f.control_offset + shape.prefill_chunk);
        f.window = original;
        auto& w  = *f.window;
        w.ids.assign(original.ids.begin() + f.control_offset, original.ids.begin() + end);
        w.main_frontier += f.control_offset;
        w.mtp_frontier = mtp_history_frontier(w.main_frontier);
        if (f.control_offset)
            for (auto& [rank, s] : w.states) s.source = s.destination;
        rows       = {&w, 1};
        f.physical = {{1, 1}, {0}, {}};
        f.physical.owners.resize(parameters.profile->local.size);
        for (std::uint32_t rank = 0; rank < f.physical.owners.size(); ++rank)
            f.physical.owners[rank] = {rank > w.attention_rank ? 1U : 0U,
                                       rank == w.attention_rank ? 1U : 0U};
    } else
        f.physical = work_batch(work);
    f.tokens = rows.front().ids.size();
    prepare_inputs(rows, f.physical, std::max(1U, f.tokens));
    if (parameters.mtp && f.tokens && (f.sequence || forward->kind == ForwardKind::Control))
        prepare_history_metadata(rows);
    const auto& io = frame;
    const int h = dimension(parameters.profile->config.text.hidden), b = f.batch.rows;
    f.hidden = f.sequence ? resources.tensor(io.prefill_hidden).slice(1, 0, std::max(1U, f.tokens))
                          : resources.packed(io.pending_hidden, {h, 1, b}).view({h, b});
    for (std::size_t i = 0; i < rows.size(); ++i)
        f.result.rows[control ? f.control_row : i].processed += rows[i].ids.size();
    if (!f.sequence && f.tokens == 1) {
        const bool sample = forward && forward->sample != SampleKind::None;
        if (graphs && forward && forward->kind == ForwardKind::Ordinary) {
            f.graph_route =
                resources.plan.decode_routes->select_route(f.physical.shape, shape.speculative, 0);
            const auto bucket =
                ops::select_context_bucket(std::max(1U, f.batch.attention.max_visible_tokens));
            (void)graphs->start_target(*f.graph_route, {.bucket = bucket, .head = sample});
        } else {
            execute_target(f.batch, sample, forward && forward->kind == ForwardKind::Control);
        }
        f.head_ready = sample;
        if (sample) {
            f.phase = ComputePhase::Decision;
            decide();
        } else {
            timer.record_result();
            compute_done.record(cuda.stream());
            f.phase = ComputePhase::AwaitForward;
        }
        return;
    }
    if (!f.sequence) upload_region(io.ingress, io.ingress.bytes);
    if (parameters.mtp && f.tokens) stage_conditioning(f.batch.rows);
    {
        auto visual     = forward ? prepare_vision(forward->visual, rows[0].main_frontier, f.tokens)
                                  : std::nullopt;
        f.timed_prefill = f.sequence && prefill_timer.has_value();
        if (f.timed_prefill) prefill_timer->start();
        execution::StateUpdate update = execution::StepUpdate{};
        if (f.sequence) {
            const auto s = rows[0].states.at(parameters.profile->local.rank);
            update       = execution::SequenceUpdate{s.source, s.destination};
        }
        main_forward(f.batch, update, f.hidden, std::move(visual));
        if (parameters.mtp)
            align_history(
                f.batch, f.hidden,
                std::max(1U, f.tokens - (rows.size() == 1 && !rows[0].main_frontier ? 1 : 0)));
        if (parameters.draft)
            append_dflash_context(f.batch, f.batch.valid_columns, {f.tokens, f.tokens});
        if (f.sequence) {
            auto last         = f.hidden.slice(1, f.tokens - 1, 1);
            const auto offset = rows[0].states.at(parameters.profile->local.rank).destination;
            auto destination =
                resources.states.image(rows[0].attention_rank == parameters.profile->local.rank)
                    .tensor(resources.plan.state.hidden, offset);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {last.data, last.bytes(), cuda.device()}, cuda);
        } else
            ops::scatter(f.hidden, resources.states.hidden(f.batch.destination_offsets),
                         f.batch.live_rows, cuda);
        if (f.timed_prefill) prefill_timer->record_stop();
    }
    if (forward && forward->sample != SampleKind::None) {
        f.phase = ComputePhase::Decision;
        decide();
    } else {
        timer.record_result();
        compute_done.record(cuda.stream());
        f.phase = ComputePhase::AwaitForward;
    }
}

void RankProgramImpl::forward_packed_prefill(const ForwardWork& work) {
    auto& f    = *compute;
    f.physical = work.batch;
    prepare_packed_inputs(work);
    const auto& packed = *f.packed;
    const auto& layout = *frame.packed_prefill;
    const int h        = dimension(parameters.profile->config.text.hidden);
    if (f.tokens) {
        f.hidden        = resources.tensor(frame.prefill_hidden).slice(1, 0, f.tokens);
        f.timed_prefill = prefill_timer.has_value();
        if (f.timed_prefill) prefill_timer->start();
        main_forward(f.batch, packed.update, f.hidden);
        if (parameters.mtp) align_packed_history(f.batch, packed, f.hidden);
        if (parameters.draft) append_packed_dflash_context(f.batch, packed.update.ragged);

        ops::scatter_ragged_last_columns(f.hidden, packed.update.ragged,
                                         resources.states.hidden(f.batch.destination_offsets),
                                         cuda);
        if (f.timed_prefill) prefill_timer->record_stop();
    }
    const int sampled = extent(packed.sampled_members.size());
    if (sampled) {
        f.sampled_hidden = resources.packed(frame.pending_hidden, {h, sampled});
        ops::gather(resources.states.hidden(row(layout.sampled_offsets, sampled)),
                    row(frame.rows.live_rows, sampled), f.sampled_hidden, cuda);
        f.phase = ComputePhase::Decision;
        decide();
    } else {
        timer.record_result();
        compute_done.record(cuda.stream());
        f.phase = ComputePhase::AwaitForward;
    }
}

execution::TextRun RankProgramImpl::text_run(const execution::BatchView& call,
                                             execution::StateUpdate update) {
    const auto& io     = frame;
    const auto records = [&](const TensorRegion& region) {
        return std::holds_alternative<execution::RecordWindow>(update)
                   ? resources.raw_record(region, call.tokens, call.attention.rows)
                   : Tensor{};
    };
    return execution::TextRun(
        parameters,
        {resources.workspace, resources.states, resources.main, records(io.main_records.raw_keys),
         records(io.main_records.raw_gates), parallel ? &*parallel : nullptr,
         parameters.draft
             ? draft_features(call, std::holds_alternative<execution::PackedPrefillUpdate>(update))
             : Tensor{}},
        call, update, cuda);
}

void RankProgramImpl::main_forward(const execution::BatchView& call, execution::StateUpdate update,
                                   Tensor hidden,
                                   std::optional<models::vision::EmbeddingView> visual,
                                   const execution::MlaSubmit& submit) {
    auto run = text_run(call, update);
    run.stem(std::move(visual));
    for (std::uint32_t i = 0; i < parameters.text.layers.size(); ++i) {
        nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime, i);
        run.local_mixer(i, submit);
        run.finish_mixer(i);
        run.ffn(i);
        run.finish_layer(i);
    }
    const auto& p       = *parameters.profile;
    const auto& columns = run.columns();
    auto output         = columns.split()
                              ? resources.tensor(frame.prefill_hidden).slice(1, 0, columns.padded)
                              : hidden.view({dimension(p.config.text.hidden), columns.columns});
    run.finish(output);
}

void RankProgramImpl::project_head(Tensor hidden) {
    auto logits = resources.packed(
        *frame.logits, {dimension(parameters.profile->local.vocabulary.size), hidden.ne[1]});
    ops::linear(hidden, parameters.text.head.weight, logits, parameters.text.head.policy,
                resources.workspace, cuda);
}

void RankProgramImpl::finish_forward() {
    auto& f = *compute;
    if (f.timed_prefill) {
        f.gpu.add_prefill(
            static_cast<std::uint64_t>(double(prefill_timer->elapsed_ms()) * 1e6 + 0.5));
        f.timed_prefill = false;
    }
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (f.window) {
        f.control_offset += f.tokens;
        if (f.control_offset == forward->rows[f.control_row].ids.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
        f.window.reset();
        if (f.control_row < forward->rows.size()) {
            f.phase = ComputePhase::Forward;
            return;
        }
    }
    if (std::holds_alternative<ScoreWork>(f.request.work.operation)) {
        f.phase = ComputePhase::Score;
        return;
    }
    if (forward->sample != SampleKind::None) {
        f.phase = ComputePhase::Decision;
        return;
    }
    if (forward->kind == ForwardKind::Control) {
        // Context restoration and forced tokens update the same per-residency penalty ledger.
        f.control_row    = 0;
        f.control_offset = 0;
        f.counts_pending = true;
        control_counts();
        return;
    }
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

void RankProgramImpl::decide() {
    auto& f           = *compute;
    const bool verify = std::holds_alternative<SpeculativeWork>(f.request.work.operation);
    if (verify && !f.decision) request_decision();
    if (!f.decision) return;
    upload_sampling();
    const int b = f.packed ? extent(f.sampling_members().size()) : f.batch.rows;
    if (f.graph_route) {
        graphs->decide(*f.graph_route);
    } else if (verify) {
        const auto& geometry = *std::get<SpeculativeWork>(f.request.work.operation).speculative;
        accept_speculative(b, f.batch.tokens, static_cast<int>(geometry.head_columns));
        copy_result(b, true);
    } else {
        if (!f.head_ready) {
            const auto hidden = f.packed     ? f.sampled_hidden
                                : f.sequence ? f.hidden.slice(1, f.tokens - 1, 1)
                                             : f.hidden;
            project_head(hidden);
        }
        const auto kind = std::get<ForwardWork>(f.request.work.operation).sample;
        sample(b, kind);
        copy_result(b, false, kind);
    }
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

void RankProgramImpl::result() {
    auto& f = *compute;
    collect_vision_timing();
    if (f.timed_prefill) {
        f.gpu.add_prefill(
            static_cast<std::uint64_t>(double(prefill_timer->elapsed_ms()) * 1e6 + 0.5));
        f.timed_prefill = false;
    }
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (forward && forward->sample != SampleKind::None) {
        const auto* ids    = static_cast<const TokenId*>(resources.egress().data);
        const auto* top1   = forward->sample == SampleKind::NextToken && frame.top1_probabilities
                                 ? static_cast<const float*>(output(*frame.top1_probabilities).data)
                                 : nullptr;
        const auto members = f.sampling_members();
        for (std::size_t b = 0; b < members.size(); ++b) {
            auto& row  = f.result.rows[members[b]];
            row.tokens = {ids[b]};
            if (top1) row.target_top1_probability = top1[b];
        }
    }
    if (const auto* speculative = std::get_if<SpeculativeWork>(&f.request.work.operation)) {
        const auto& io         = frame;
        const auto* ids        = static_cast<const TokenId*>(output(*io.sampled_ids).data);
        const auto* counts     = static_cast<const std::int32_t*>(output(*io.output_counts).data);
        const auto* confidence = static_cast<const float*>(output(*io.draft_confidences).data);
        const auto& route      = *speculative->speculative;
        for (std::size_t b = 0; b < f.physical.members.size(); ++b) {
            const auto logical = f.physical.members[b];
            if (counts[b] < 1 || std::uint32_t(counts[b]) > speculative->draft_extents[logical] + 1)
                throw std::logic_error("GLM5Next acceptance returned an invalid prefix");
            auto& r           = f.result.rows[logical];
            r.processed       = speculative->draft_extents[logical] + 1;
            r.drafts          = speculative->draft_extents[logical];
            r.accepted_drafts = counts[b] - 1;
            for (std::uint32_t step = 0; step < r.drafts; ++step)
                r.draft_confidences.push_back(
                    confidence[route.mtp_steps ? step * route.physical_rows + b
                                               : b * route.head_columns + step]);
            r.tokens.assign(ids + b * speculative->speculative->verify_width,
                            ids + b * speculative->speculative->verify_width + counts[b]);
        }
    }
    timer.observe_result(f.gpu);
    if (ExecutionTypes::describe_work(f.request.work).kind == ExecutionKind::Decode)
        timer.observe_decode(f.gpu,
                             std::holds_alternative<SpeculativeWork>(f.request.work.operation));
    if (resources.compute_actions->occupied()) resources.compute_actions->finish();
    events.emplace_back(runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(f.result)});
    f.phase = ComputePhase::PendingCommit;
}

void RankProgramImpl::advance_compute() {
    if (!compute) return;
    auto& f = *compute;
    switch (f.phase) {
    case ComputePhase::Forward:
        if (std::holds_alternative<SpeculativeWork>(f.request.work.operation))
            prepare_speculative();
        else
            forward();
        break;
    case ComputePhase::AwaitForward:
        if (compute_done.ready()) finish_forward();
        break;
    case ComputePhase::Decision:
        decide();
        break;
    case ComputePhase::AwaitResult:
        if (compute_done.ready()) {
            if (f.counts_pending)
                control_counts();
            else
                result();
        }
        break;
    case ComputePhase::Score:
        score();
        break;
    case ComputePhase::AwaitScore:
        if (compute_done.ready()) {
            const auto* words = static_cast<const float*>(resources.egress().data);
            f.logprobs.insert(f.logprobs.end(), words, words + f.score_count);
            f.score_at += f.score_count;
            f.phase = ComputePhase::Score;
        }
        break;
    case ComputePhase::PendingCommit:
        break;
    case ComputePhase::Retiring:
        if (compute_done.ready()) {
            if (resources.compute_actions->occupied()) resources.compute_actions->finish();
            collect_vision_timing();
            if (const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
                forward && forward->visual) {
                const auto& use = forward->visual->window;
                const auto& row = forward->rows.front();
                if (f.release_vision || row.main_frontier + row.ids.size() >= use.plan->consume_end)
                    resources.vision->release(use.id);
            }
            auto observed = observation();
            observed.path = f.graph_route ? ExecutionPath::Graph : ExecutionPath::Eager;
            timer.observe_retirement(f.gpu);
            observed.gpu       = f.gpu;
            observed.attention = {ExecutionTypes::describe_work(f.request.work).kind, f.attention};
            const auto ticket  = f.request.ticket;
            const auto command = f.retirement;
            compute.reset();
            events.emplace_back(runtime::DeviceRetired{ticket, command, std::move(observed)});
        }
        break;
    }
}
} // namespace ninfer::models::glm5next::detail
