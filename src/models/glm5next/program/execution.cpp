#include "models/glm5next/program/internal.h"
#include "core/transfer.h"
#include "core/nvtx.h"
#include "ninfer/ops/indexing.h"

namespace ninfer::models::glm5next::detail {
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
        if (f.control_offset)
            for (auto& [rank, s] : w.states) s.source = s.destination;
        rows       = {&w, 1};
        f.physical = plan_batch(rows, parameters.profile->local.size);
    } else
        f.physical = work.batch;
    f.tokens = rows.front().ids.size();
    upload(rows, f.physical, std::max(1U, f.tokens));
    const auto& io = resources.plan.frame;
    const int h = dimension(parameters.profile->config.text.hidden), b = rows.size();
    f.hidden = f.sequence ? resources.tensor(io.prefill_hidden).slice(1, 0, std::max(1U, f.tokens))
                          : resources.packed(io.pending_hidden, {h, 1, b}).view({h, b});
    if (!f.tokens) {
        const auto offset = rows[0].states.at(parameters.profile->local.rank).source;
        const auto source =
            resources.states.image(rows[0].attention_rank == parameters.profile->local.rank)
                .tensor(resources.plan.state.hidden, offset);
        copy_bytes_async({f.hidden.data, f.hidden.bytes(), cuda.device()},
                         {source.data, source.bytes(), cuda.device()}, cuda);
    } else {
        f.timed_prefill = f.sequence && prefill_timer.has_value();
        if (f.timed_prefill) prefill_timer->start();
        execution::StateUpdate update = execution::StepUpdate{};
        if (f.sequence) {
            const auto s = rows[0].states.at(parameters.profile->local.rank);
            update       = execution::SequenceUpdate{s.source, s.destination};
        }
        execution::TextRun run(parameters,
                               {resources.workspace, resources.states, resources.main,
                                resources.tensor(io.main_records.raw_keys),
                                resources.tensor(io.main_records.raw_gates),
                                parallel ? &*parallel : nullptr},
                               f.batch, update, cuda);
        run.stem();
        for (std::uint32_t i = 0; i < parameters.text.layers.size(); ++i) {
            nvtx::ScopedRange range(nvtx::Name::TextLayer, nvtx::Category::Runtime, i);
            run.local_mixer(i);
            run.finish_mixer(i);
            run.ffn(i);
            run.finish_layer(i);
        }
        run.finish(f.hidden);
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
    for (std::size_t i = 0; i < rows.size(); ++i)
        f.result.rows[control ? f.control_row : i].processed += rows[i].ids.size();
    if (forward && forward->sample != SampleKind::None) {
        f.phase = ComputePhase::Decision;
        decide();
    } else {
        timer.record_result();
        compute_done.record(cuda.stream());
        f.phase = ComputePhase::AwaitForward;
    }
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
    auto& f = *compute;
    if (!f.decision) return;
    upload_sampling();
    const auto& io = resources.plan.frame;
    const int b    = work_rows(f.request.work).size();
    auto hidden    = f.sequence ? f.hidden.slice(1, f.tokens ? f.tokens - 1 : 0, 1) : f.hidden;
    auto logits =
        resources.packed(*io.logits, {dimension(parameters.profile->local.vocabulary.size), b});
    ops::linear(hidden, parameters.text.head.weight, logits, parameters.text.head.policy,
                resources.workspace, cuda);
    sample();
    auto ids = resources.packed(*io.sampled_ids, {b});
    copy_bytes_async(resources.egress().subspan(0, ids.bytes()),
                     {ids.data, ids.bytes(), cuda.device()}, cuda);
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

void RankProgramImpl::result() {
    auto& f = *compute;
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
