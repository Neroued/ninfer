#include "models/glm5next/program/internal.h"

#include "core/nvtx.h"
#include "core/transfer.h"
#include "models/glm5next/state/continuation.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/sequence.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
void RankProgramImpl::execute_round(const execution::BatchView& main,
                                    std::span<const std::uint32_t> draft_bounds, bool head,
                                    bool read_drafts, const execution::MlaSubmit& submit,
                                    RoundSections* sections) {
    const auto& io = resources.plan.frame;
    const int g = main.rows, c = main.attention.rows, h = parameters.profile->config.text.hidden;
    const bool verify = !draft_bounds.empty();
    auto hidden       = resources.packed(io.pending_hidden, {h, main.tokens, g});
    if (parameters.mtp) stage_conditioning(g);
    if (verify) {
        const auto& m = *io.mtp;
        auto drafts   = resources.packed(m.drafts, {dimension(shape.draft_tokens), g});
        CUDA_CHECK(cudaMemsetAsync(drafts.data, 0, drafts.bytes(), cuda.stream()));
        seed_mtp_tail(main, false);
        {
            nvtx::ScopedRange range(nvtx::Name::DecodeMtpDraft, nvtx::Category::Mtp);
            for (std::uint32_t step = 0; step < draft_bounds.size(); ++step)
                draft_forward(step, batch(m.draft_steps[step], 1, g, c, draft_bounds[step]),
                              submit);
        }
        const auto readback = [&] {
            copy_bytes_async(output(m.drafts).subspan(0, drafts.bytes()),
                             {drafts.data, drafts.bytes(), cuda.device()}, cuda);
        };
        if (sections) {
            sections->drafts.capture(cuda.stream(), readback);
            drafts_done.record_external(cuda.stream());
        } else if (read_drafts) {
            readback();
            drafts_done.record(cuda.stream());
        }
        auto ids = main.ids, positions = main.positions;
        ops::prepare_verify_inputs(row(m.anchors, g), drafts, row(io.rows.base_positions, g),
                                   row(m.extents, g), ids, positions, cuda);
        {
            nvtx::ScopedRange range(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp);
            main_forward(main, execution::RecordWindow{*resources.kda_records}, hidden, {}, submit);
        }
    } else {
        main_forward(main, execution::StepUpdate{}, hidden, {}, submit);
        if (parameters.mtp) align_history(main, hidden, 1);
        ops::scatter(hidden.view({h, g}), resources.states.hidden(main.destination_offsets), cuda);
    }
    const auto project = [&] { project_head(hidden.view({h, main.tokens * g})); };
    if (sections)
        sections->head.capture(cuda.stream(), project);
    else if (head)
        project();
}

void RankProgramImpl::forward() {
    auto& f             = *compute;
    const auto& work    = f.request.work;
    auto rows           = work_rows(work);
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    const bool control =
        forward && forward->kind == ForwardKind::Control && rows.front().ids.size() > 1;
    f.sequence = std::holds_alternative<ScoreWork>(work.operation) || control ||
                 (forward &&
                  (forward->kind == ForwardKind::Prefill || forward->kind == ForwardKind::Replay));
    if (control) {
        const auto& original = rows[f.control_row];
        auto end =
            std::min<std::uint32_t>(original.ids.size(), f.control_offset + shape.prefill_chunk);
        for (auto boundary : original.window_ends)
            if (boundary > f.control_offset) {
                end = std::min(end, boundary);
                break;
            }
        f.window = original;
        auto& w  = *f.window;
        w.ids.assign(original.ids.begin() + f.control_offset, original.ids.begin() + end);
        w.main_frontier += f.control_offset;
        w.mtp_frontier = mtp_history_frontier(w.main_frontier);
        if (f.control_offset)
            for (auto& [rank, s] : w.states) s.source = s.destination;
        rows       = {&w, 1};
        f.physical = plan_batch(rows, parameters.profile->local.size);
    } else
        f.physical = work.batch;
    f.tokens = rows.front().ids.size();
    upload(rows, f.physical, std::max(1U, f.tokens));
    if (parameters.mtp && f.tokens) { prepare_history_metadata(rows); }
    const auto& io = resources.plan.frame;
    const int h = dimension(parameters.profile->config.text.hidden), b = rows.size();
    f.hidden = f.sequence ? resources.tensor(io.prefill_hidden).slice(1, 0, std::max(1U, f.tokens))
                          : resources.packed(io.pending_hidden, {h, 1, b}).view({h, b});
    for (std::size_t i = 0; i < rows.size(); ++i)
        f.result.rows[control ? f.control_row : i].processed += rows[i].ids.size();
    if (!f.sequence && f.tokens == 1) {
        const bool sample = forward && forward->sample != SampleKind::None;
        f.graph_round     = graphs != nullptr;
        if (graphs)
            graphs->replay(f.batch, {}, sample, false);
        else
            execute_round(f.batch, {}, sample, false);
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
    if (parameters.mtp && f.tokens) stage_conditioning(rows.size());
    if (!f.tokens) {
        const auto offset = rows[0].states.at(parameters.profile->local.rank).source;
        const auto source =
            resources.states.image(rows[0].attention_rank == parameters.profile->local.rank)
                .tensor(resources.plan.state.hidden, offset);
        copy_bytes_async({f.hidden.data, f.hidden.bytes(), cuda.device()},
                         {source.data, source.bytes(), cuda.device()}, cuda);
    } else {
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
        if (f.sequence) {
            auto last         = f.hidden.slice(1, f.tokens - 1, 1);
            const auto offset = rows[0].states.at(parameters.profile->local.rank).destination;
            auto destination =
                resources.states.image(rows[0].attention_rank == parameters.profile->local.rank)
                    .tensor(resources.plan.state.hidden, offset);
            copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                             {last.data, last.bytes(), cuda.device()}, cuda);
        } else
            ops::scatter(f.hidden, resources.states.hidden(f.batch.destination_offsets), cuda);
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

execution::TextRun RankProgramImpl::text_run(const execution::BatchView& call,
                                             execution::StateUpdate update) {
    const auto& io = resources.plan.frame;
    return execution::TextRun(parameters,
                              {resources.workspace, resources.states, resources.main,
                               resources.tensor(io.main_records.raw_keys),
                               resources.tensor(io.main_records.raw_gates),
                               parallel ? &*parallel : nullptr},
                              call, update, cuda);
}

void RankProgramImpl::main_forward(const execution::BatchView& call, execution::StateUpdate update,
                                   Tensor hidden,
                                   std::optional<models::vision::EmbeddingView> visual,
                                   const execution::MlaSubmit& submit) {
    localize(call);
    auto run = text_run(call, update);
    run.stem(std::move(visual));
    for (std::uint32_t i = 0; i < parameters.text.layers.size(); ++i) {
        nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime, i);
        run.local_mixer(i, submit);
        run.finish_mixer(i);
        run.ffn(i);
        run.finish_layer(i);
    }
    auto output =
        hidden.view({dimension(parameters.profile->config.text.hidden), call.tokens * call.rows});
    run.finish(output);
}

void RankProgramImpl::project_head(Tensor hidden) {
    auto logits =
        resources.packed(*resources.plan.frame.logits,
                         {dimension(parameters.profile->local.vocabulary.size), hidden.ne[1]});
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
    auto& f        = *compute;
    const bool mtp = std::holds_alternative<MtpWork>(f.request.work.operation);
    if (mtp && !f.decision) request_decision();
    if (!f.decision) return;
    upload_sampling();
    const auto rows = work_rows(f.request.work);
    const int b     = rows.size();
    if (f.graph_round) {
        graphs->decide(b, mtp);
    } else if (mtp) {
        accept_mtp(b);
        copy_result(b, true);
    } else {
        if (!f.head_ready) {
            auto hidden = f.sequence ? f.hidden.slice(1, f.tokens ? f.tokens - 1 : 0, 1) : f.hidden;
            project_head(hidden);
        }
        sample(b, std::get<ForwardWork>(f.request.work.operation).sample);
        copy_result(b, false);
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
        const auto* ids = static_cast<const TokenId*>(resources.egress().data);
        for (std::size_t b = 0; b < f.physical.members.size(); ++b)
            f.result.rows[f.physical.members[b]].tokens = {ids[b]};
    }
    if (const auto* mtp = std::get_if<MtpWork>(&f.request.work.operation)) {
        const auto& io     = resources.plan.frame;
        const auto* ids    = static_cast<const TokenId*>(output(*io.sampled_ids).data);
        const auto* counts = static_cast<const std::int32_t*>(output(*io.output_counts).data);
        const auto* accepted =
            static_cast<const std::int32_t*>(output(io.mtp->accepted_drafts).data);
        for (std::size_t b = 0; b < f.physical.members.size(); ++b) {
            const auto logical = f.physical.members[b];
            if (counts[b] < 1 || std::uint32_t(counts[b]) > mtp->draft_extents[logical] + 1 ||
                accepted[b] != counts[b] - 1)
                throw std::logic_error("GLM5Next acceptance returned an invalid prefix");
            auto& r           = f.result.rows[logical];
            r.processed       = mtp->draft_extents[logical] + 1;
            r.drafts          = mtp->draft_extents[logical];
            r.accepted_drafts = accepted[b];
            r.tokens.assign(ids + b * (shape.draft_tokens + 1),
                            ids + b * (shape.draft_tokens + 1) + counts[b]);
        }
    }
    timer.observe_result(f.gpu);
    if (resources.compute_actions->occupied()) resources.compute_actions->finish();
    events.emplace_back(runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(f.result)});
    f.phase = ComputePhase::PendingCommit;
}

void RankProgramImpl::advance_compute() {
    if (!compute) return;
    auto& f = *compute;
    switch (f.phase) {
    case ComputePhase::Forward:
        if (std::holds_alternative<MtpWork>(f.request.work.operation))
            prepare_mtp();
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
            timer.observe_retirement(f.gpu);
            observed.gpu       = f.gpu;
            const auto ticket  = f.request.ticket;
            const auto command = f.retirement;
            compute.reset();
            events.emplace_back(runtime::DeviceRetired{ticket, command, std::move(observed)});
        }
        break;
    }
}
} // namespace ninfer::models::glm5next::detail
