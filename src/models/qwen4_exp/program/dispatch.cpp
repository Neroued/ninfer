#include "models/qwen4_exp/program/internal.h"
#include "core/nvtx.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::start(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    if (request.ticket.frame.kind == runtime::FrameKind::Context) {
        if (context) throw std::logic_error("Qwen4Exp context frame is occupied");
        context.emplace(ContextFrame{std::move(request), false});
        resources.context_actions->submit(context->request.work.actions);
        advance_context();
        return;
    }
    if (compute) throw std::logic_error("Qwen4Exp compute frame is occupied");
    compute.emplace();
    auto& f      = *compute;
    f.request    = std::move(request);
    f.retirement = f.request.command;
    std::visit(
        [&](const auto& work) {
            using T = std::decay_t<decltype(work)>;
            if constexpr (std::is_same_v<T, ForwardWork> || std::is_same_v<T, MtpWork>)
                f.rows = work.rows;
            else if constexpr (std::is_same_v<T, ScoreWork>)
                f.rows = {&work.row, 1};
        },
        f.request.work.operation);
    resources.compute_actions->submit(f.request.work.actions);
    advance_compute();
}

void RankProgramImpl::advance_compute() {
    if (!compute) return;
    auto& f = *compute;
    for (;;) {
        switch (f.phase) {
        case ComputePhase::Prepare:
            if (std::holds_alternative<MtpWork>(f.request.work.operation))
                prepare_mtp();
            else
                prepare_forward();
            if (f.phase == ComputePhase::Prepare) return;
            break;
        case ComputePhase::Prefix:
            if (f.graph ? !f.graph->advance_prefix(cuda)
                        : !advance_layers(parameters.profile->config.text.ple.layer))
                return;
            f.phase = ComputePhase::AwaitPle;
            break;
        case ComputePhase::AwaitPle: {
            request_decision();
            const auto embedding = ple.try_consume(*f.ple);
            if (!embedding) return;
            request_decision();
            if (f.verify) {
                for (std::size_t b = 0; b < f.rows.size(); ++b) {
                    const auto input = ple.input(*f.ple, b);
                    std::copy(input.begin(), input.end(), f.verify->inputs.begin() + b * f.tokens);
                }
            }
            if (f.graph) {
                std::optional<nvtx::ScopedRange> range;
                if (f.verify) range.emplace(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp);
                if (f.graph->resident())
                    f.graph->replay(f.ple->slot, cuda);
                else
                    f.graph->consume_ple(f.ple->slot, cuda);
            } else {
                embedding->ready.wait(cuda.stream());
                f.run->consume_ple(embedding->values);
                embedding->consumed.record(cuda.stream());
            }
            ple.consumer_submitted(*f.ple);
            if (const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
                forward && forward->prefetch) {
                const auto& next = *forward->prefetch;
                const ple::InputRow input{{next.member.sequence.value,
                                           next.member.sequence.generation, next.frontier,
                                           next.raw_tail},
                                          next.ids};
                ple.prefetch(next.ids.size(), {&input, 1});
            }
            f.phase =
                f.graph && f.graph->resident() ? ComputePhase::AfterMain : ComputePhase::Suffix;
            break;
        }
        case ComputePhase::Suffix:
            if (f.graph) {
                if (!f.graph->advance_suffix(cuda)) return;
            } else {
                if (!advance_layers(parameters.text.layers.size())) return;
                Tensor logits;
                const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
                if (f.verify || (forward && forward->sample == SampleKind::NextToken))
                    logits = packed_tensor(*resources.plan.persistent.buffers.logits,
                                           {dimension(parameters.profile->config.text.vocabulary),
                                            static_cast<int>(f.tokens),
                                            static_cast<int>(f.rows.size())});
                f.run->finish({f.sample, f.conditioning}, logits);
            }
            f.phase = ComputePhase::AfterMain;
            break;
        case ComputePhase::AfterMain:
            after_main();
            break;
        case ComputePhase::Decision:
            prepare_decision();
            if (f.phase == ComputePhase::Decision) return;
            break;
        case ComputePhase::AwaitResult:
            if (!compute_done.ready()) return;
            if (const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
                forward && forward->kind == ForwardKind::Control)
                finish_control_window();
            else
                finish_result();
            return;
        case ComputePhase::Score:
            score_next();
            if (f.phase == ComputePhase::Score) return;
            break;
        case ComputePhase::AwaitScore:
            if (!compute_done.ready()) return;
            {
                const auto* values = static_cast<const float*>(resources.egress().data);
                f.logprobs.insert(f.logprobs.end(), values, values + f.score_count);
                f.score_at += f.score_count;
                f.phase = ComputePhase::Score;
            }
            break;
        case ComputePhase::ControlCounts:
            enqueue_control_counts();
            break;
        case ComputePhase::AwaitControl:
            if (!compute_done.ready()) return;
            if (f.control_row != std::get<ForwardWork>(f.request.work.operation).rows.size()) {
                f.phase = ComputePhase::ControlCounts;
                break;
            }
            f.result.timing = f.timing;
            events.emplace_back(
                runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(f.result)});
            if (resources.compute_actions->occupied()) resources.compute_actions->finish();
            f.phase = ComputePhase::PendingCommit;
            return;
        case ComputePhase::PendingCommit:
            return;
        case ComputePhase::Retiring:
            if (!compute_done.ready() || !ple.retired(f.request.ticket)) return;
            retire_compute();
            return;
        }
    }
}

void RankProgramImpl::finish_result() {
    auto& f      = *compute;
    auto& result = f.result;
    result.rows.resize(f.rows.size());
    const auto& buffers          = resources.plan.persistent.buffers;
    const auto* ids              = static_cast<const TokenId*>(resources.egress().data);
    const auto* forward          = std::get_if<ForwardWork>(&f.request.work.operation);
    const bool sampled           = f.verify || (forward && forward->sample != SampleKind::None);
    const std::int32_t* counts   = nullptr;
    const std::int32_t* accepted = nullptr;
    if (f.verify) {
        counts = reinterpret_cast<const std::int32_t*>(
            resources.egress()
                .subspan(buffers.output_counts->region.offset - buffers.sampled_ids->region.offset)
                .data);
        accepted = reinterpret_cast<const std::int32_t*>(
            resources.egress()
                .subspan(buffers.accepted_drafts->region.offset -
                         buffers.sampled_ids->region.offset)
                .data);
    }
    for (std::size_t b = 0; b < f.rows.size(); ++b) {
        auto& row     = result.rows[b];
        row.processed = f.verify ? 0 : f.rows[b].ids.size();
        row.drafts    = f.verify ? f.verify->extents[b] : 0;
        if (f.verify)
            row.input_tokens.assign(f.verify->inputs.begin() + b * f.verify->width,
                                    f.verify->inputs.begin() + b * f.verify->width + row.drafts +
                                        1);
        if (sampled) {
            const auto count = counts ? counts[b] : 1;
            if (count < 1 || static_cast<std::uint32_t>(count) > row.drafts + 1)
                throw std::logic_error("Qwen4Exp generated prefix exceeds its licensed extent");
            const auto stride = f.verify ? f.verify->width : 1;
            row.tokens.assign(ids + stride * b, ids + stride * b + count);
            row.accepted_drafts = accepted ? accepted[b] : 0;
            row.dead_positions  = f.dead_positions[b];
        }
        row.vision_seconds = collect_vision_timing(binding(f.rows[b].member.residency),
                                                   f.rows[b].main_frontier + row.processed);
    }
    if (f.timer_started)
        f.timing.gpu_elapsed_ns +=
            static_cast<std::uint64_t>(double(prefill_timer.elapsed_ms()) * 1.0e6 + 0.5);
    result.timing = f.timing;
    events.emplace_back(runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(result)});
    if (resources.compute_actions->occupied()) resources.compute_actions->finish();
    f.phase = ComputePhase::PendingCommit;
}

void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& commit) {
    if (!compute || compute->request.ticket != ticket)
        throw std::logic_error("Qwen4Exp commit refers to an inactive execution");
    auto& f = *compute;
    if (f.verify) commit_verify(commit);
    resources.compute_actions->submit(commit.actions);
    compute_done.record(cuda.stream());
    f.retirement = command;
    f.phase      = ComputePhase::Retiring;
}

void RankProgramImpl::discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
    if (!compute || compute->request.ticket != ticket)
        throw std::logic_error("Qwen4Exp discard refers to an inactive execution");
    std::erase_if(events, [&](const auto& event) {
        return std::visit(
            [&](const auto& value) {
                using Event = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<Event, runtime::DecisionNeeded> ||
                              std::is_same_v<Event, runtime::ResultReady<ExecutionTypes>>)
                    return value.ticket == ticket;
                return false;
            },
            event);
    });
    auto& f      = *compute;
    f.retirement = command;
    ple.cancel(ticket);
    if (f.graph) f.graph->cancel();
    if (resources.experts) resources.experts->cancel();
    f.run.reset();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::Retiring;
}

void RankProgramImpl::retire_compute() {
    const auto ticket  = compute->request.ticket;
    const auto command = compute->retirement;
    if (resources.compute_actions->occupied()) resources.compute_actions->finish();
    compute.reset();
    events.emplace_back(runtime::DeviceRetired{ticket, command});
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgramImpl::poll() {
    ple.progress();
    advance_context();
    advance_compute();
    if (events.empty()) return {};
    auto event = std::move(events.front());
    events.pop_front();
    return event;
}

void RankProgramImpl::drain() {
    if (drained) return;
    std::exception_ptr failure;
    const auto attempt = [&](auto&& action) {
        try {
            action();
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    };
    if (compute) {
        attempt([&] { ple.cancel(compute->request.ticket); });
        if (compute->graph) attempt([&] { compute->graph->cancel(); });
    }
    if (resources.experts) attempt([&] { resources.experts->cancel(); });
    attempt([&] { ple.drain(); });
    attempt([&] { resources.synchronize(); });
    compute.reset();
    context.reset();
    events.clear();
    for (auto& resident : bindings) {
        if (resident) attempt([&] { retire_vision(*resident); });
        resident.reset();
    }
    drained = true;
    if (failure) std::rethrow_exception(failure);
}

} // namespace ninfer::models::qwen4_exp::detail
