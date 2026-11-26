#include "models/inkling2/program/internal.h"

#include "core/nvtx.h"
#include "core/transfer.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/logit_softcap.h"

#include <algorithm>

namespace ninfer::models::inkling2::detail {

void RankProgramImpl::execute_target(const execution::BatchView& main, bool head) {
    auto hidden = target_hidden(main);
    begin_target(main, /*capture=*/false);
    target_forward(main, hidden);
    if (head) project_head(hidden.view({hidden.ne[0], main.tokens * main.rows}));
    timer.end_target();
}

void RankProgramImpl::begin_target(const execution::BatchView& main, bool capture) {
    if (main.tokens == 1) upload_region(frame.ingress, frame.ingress.bytes);
    timer.begin_target(capture);
}

Tensor RankProgramImpl::target_hidden(const execution::BatchView& main) {
    const auto h = dimension(parameters.profile->config.text.hidden);
    return resources.packed(frame.pending_hidden, {h, main.tokens, main.rows});
}

void RankProgramImpl::target_forward(const execution::BatchView& main, Tensor hidden,
                                     const execution::FullAttentionSubmit& submit) {
    main_forward(main, execution::StepUpdate{}, hidden, submit);
    ops::scatter(hidden.view({hidden.ne[0], main.rows}),
                 resources.states.hidden(main.destination_offsets), main.live_rows, cuda);
}

void RankProgramImpl::forward() {
    auto& f             = *compute;
    const auto& work    = f.request.work;
    auto rows           = work_rows(work);
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    if (forward && forward->kind == ForwardKind::Prefill) {
        forward_packed_prefill(*forward);
        return;
    }
    const bool control =
        forward && forward->kind == ForwardKind::Control && rows.front().ids.size() > 1;
    f.sequence = std::holds_alternative<ScoreWork>(work.operation) || control ||
                 (forward && forward->kind == ForwardKind::Replay);
    if (control) {
        const auto& original = rows[f.control_row];
        auto end =
            std::min<std::uint32_t>(original.ids.size(), f.control_offset + shape.prefill_chunk);
        f.window = original;
        auto& w  = *f.window;
        w.ids.assign(original.ids.begin() + f.control_offset, original.ids.begin() + end);
        w.main_frontier += f.control_offset;
        if (f.control_offset)
            for (auto& [rank, s] : w.states) s.source = s.destination;
        rows       = {&w, 1};
        f.physical = {{1, 1}, {0}, {{0, 1}}};
    } else {
        f.physical = work_batch(work);
    }
    f.tokens = static_cast<std::uint32_t>(rows.front().ids.size());
    prepare_inputs(rows, f.physical, std::max(1U, f.tokens));
    const int h = dimension(parameters.profile->config.text.hidden);
    const int b = f.batch.rows;
    f.hidden    = f.sequence
                      ? resources.tensor(frame.prefill_hidden).slice(1, 0, std::max(1U, f.tokens))
                      : resources.packed(frame.pending_hidden, {h, 1, b}).view({h, b});
    for (std::size_t i = 0; i < rows.size(); ++i)
        f.result.rows[control ? f.control_row : i].processed +=
            static_cast<std::uint32_t>(rows[i].ids.size());

    if (!f.sequence && f.tokens == 1) {
        const bool do_sample = forward && forward->sample != SampleKind::None;
        if (graphs && forward && forward->kind == ForwardKind::Ordinary) {
            f.graph_route =
                resources.plan.decode_routes->select_route(f.physical.shape, shape.speculative, 0);
            const auto bucket =
                ops::select_context_bucket(std::max(1U, f.batch.max_visible_tokens));
            (void)graphs->start_target(*f.graph_route, {.bucket = bucket, .head = do_sample});
        } else {
            execute_target(f.batch, do_sample);
        }
        f.head_ready = do_sample;
        if (do_sample) {
            f.phase = ComputePhase::Decision;
            decide();
        } else {
            timer.record_result();
            compute_done.record(cuda.stream());
            f.phase = ComputePhase::AwaitForward;
        }
        return;
    }

    // Sequence ragged execution (Replay, Score, or multi-token Control)
    f.timed_prefill = f.sequence && prefill_timer.has_value();
    if (f.timed_prefill) prefill_timer->start();
    main_forward(f.batch, execution::PackedPrefillUpdate{f.sequence_ragged}, f.hidden);
    {
        auto last         = f.hidden.slice(1, static_cast<std::int32_t>(f.tokens) - 1, 1);
        const auto offset = rows[0].states.at(parameters.profile->local.rank).destination;
        auto destination  = resources.states.hidden(offset);
        copy_bytes_async({destination.data, destination.bytes(), cuda.device()},
                         {last.data, last.bytes(), cuda.device()}, cuda);
    }
    if (f.timed_prefill) prefill_timer->record_stop();
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
    const int h        = dimension(parameters.profile->config.text.hidden);
    if (f.tokens) {
        f.hidden        = resources.tensor(frame.prefill_hidden).slice(1, 0, f.tokens);
        f.timed_prefill = prefill_timer.has_value();
        if (f.timed_prefill) prefill_timer->start();
        main_forward(f.batch, packed.update, f.hidden);
        ops::scatter_ragged_last_columns(f.hidden, packed.update.ragged,
                                         resources.states.hidden(f.batch.destination_offsets),
                                         cuda);
        if (f.timed_prefill) prefill_timer->record_stop();
    }
    const int sampled = extent(packed.sampled_members.size());
    if (sampled) {
        f.sampled_hidden = resources.packed(frame.pending_hidden, {h, sampled});
        ops::gather(resources.states.hidden(row(frame.packed.sampled_offsets, sampled)),
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
    return execution::TextRun(parameters, {resources.workspace, {resources.states, resources.main}},
                              call, update, cuda);
}

void RankProgramImpl::main_forward(const execution::BatchView& call, execution::StateUpdate update,
                                   Tensor hidden, const execution::FullAttentionSubmit& submit) {
    auto run = text_run(call, update);
    run.stem();
    run.forward(hidden, submit);
}

void RankProgramImpl::project_head(Tensor hidden) {
    auto logits = resources.packed(
        *frame.logits, {dimension(parameters.profile->local.vocabulary.size), hidden.ne[1]});
    ops::linear(hidden, parameters.text.head.weight, logits, parameters.text.head.policy,
                resources.workspace, cuda);
    ops::logit_softcap(logits, parameters.profile->config.text.final_logit_softcapping, logits,
                       cuda);
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
    const int b = f.packed ? extent(f.sampling_members().size()) : f.batch.rows;
    if (f.graph_route) {
        graphs->decide(*f.graph_route);
    } else {
        if (!f.head_ready) {
            const auto hidden = f.packed     ? f.sampled_hidden
                                : f.sequence ? f.hidden.slice(1, f.tokens - 1, 1)
                                             : f.hidden;
            project_head(hidden);
        }
        const auto kind = std::get<ForwardWork>(f.request.work.operation).sample;
        sample(b, kind);
        copy_result(b);
    }
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
        const auto* ids    = static_cast<const TokenId*>(resources.egress().data);
        const auto members = f.sampling_members();
        for (std::size_t b = 0; b < members.size(); ++b) {
            f.result.rows[members[b]].tokens = {ids[b]};
        }
    }
    timer.observe_result(f.gpu);
    if (ExecutionTypes::describe_work(f.request.work).kind == ExecutionKind::Decode)
        timer.observe_decode(f.gpu, false);
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

} // namespace ninfer::models::inkling2::detail
