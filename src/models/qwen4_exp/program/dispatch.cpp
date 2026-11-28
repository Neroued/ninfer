#include "models/qwen4_exp/program/internal.h"
#include "core/nvtx.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {

void RankProgramImpl::serve_expert_pass(bool draft, std::uint32_t columns,
                                        std::span<const std::uint8_t> owners) {
    const auto serve = [&](const moe::Parameters& layer) {
        const auto stage = exchange->stage(layer.bank, dimension(columns), owners);
        moe::resident_layer(layer, nullptr, &stage, resources.workspace, cuda);
    };
    if (draft)
        serve(parameters.mtp.value().layer.moe);
    else
        for (const auto& layer : parameters.text.layers) serve(layer.moe);
}

void RankProgramImpl::start(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    if (request.ticket.frame.kind == runtime::FrameKind::Context) {
        if (context) throw std::logic_error("Qwen4Exp context frame is occupied");
        context.emplace(ContextFrame{std::move(request), false});
        resources.context_actions->submit(
            local_actions(context->request.work.actions, parameters.profile->replica_rank()));
        advance_context();
        return;
    }
    const auto index = request.ticket.frame.index;
    if (index >= retained.size() || retained[index] ||
        (compute && compute->request.ticket.frame.index == index) ||
        std::ranges::any_of(waiting, [&](const auto& r) { return r.ticket.frame.index == index; }))
        throw std::logic_error("Qwen4Exp compute frame is occupied");
    waiting.push_back(std::move(request));
    admit_compute();
}

const ComputeFrame* RankProgramImpl::find_frame(runtime::ExecutionTicket ticket) const {
    if (compute && compute->request.ticket == ticket) return compute.get();
    if (ticket.frame.index < retained.size()) {
        const auto& saved = retained[ticket.frame.index];
        if (saved && saved->request.ticket == ticket) return saved.get();
    }
    return nullptr;
}

void RankProgramImpl::admit_compute() {
    for (auto& saved : retained) {
        if (saved && saved->phase == ComputePhase::Retiring && consumers_retired(*saved))
            publish_retirement(std::move(saved));
    }
    if (compute && retained.size() > 1 && compute->phase == ComputePhase::PendingCommit) {
        const auto slot = compute->request.ticket.frame.index;
        retained[slot]  = std::move(compute);
    }
    if (compute) return;
    for (auto& saved : retained) {
        if (!saved || (!saved->deferred_commit && !saved->deferred_discard)) continue;
        compute           = std::move(saved);
        const auto ticket = compute->request.ticket;
        gpu_timer.begin_resolution();
        if (compute->deferred_discard)
            apply_discard(ticket, compute->retirement);
        else {
            auto command = std::move(*compute->deferred_commit);
            compute->deferred_commit.reset();
            apply_commit(ticket, compute->retirement, std::move(command));
        }
        return;
    }
    if (!waiting.empty()) {
        auto next = std::move(waiting.front());
        waiting.pop_front();
        submit_compute(std::move(next));
    }
}

void RankProgramImpl::submit_compute(runtime::ExecutionRequest<ExecutionTypes>&& request) {
    gpu_timer.begin();
    compute    = std::make_unique<ComputeFrame>();
    auto& f   = *compute;
    f.request = std::move(request);
    f.decision = std::move(f.request.work.initial_decision);
    if (const auto& step = f.request.work.exchange) {
        last_expert_step = step->id;
        f.main_pass      = step->draft_passes;
        if (std::holds_alternative<moe::ExpertServiceWork>(f.request.work.operation)) {
            for (std::size_t pass = 0; pass < step->columns.size(); ++pass) {
                const bool draft = pass < step->draft_passes;
                if (step->graph)
                    service_graphs->launch(draft, step->columns[pass], cuda);
                else
                    serve_expert_pass(draft, step->columns[pass], step->source_owners);
            }
            gpu_timer.record_result();
            compute_done.record(cuda.stream());
            f.phase = ComputePhase::AwaitExpertService;
            return;
        }
    }
    f.physical          = work_shape(f.request.work);
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    const auto* mtp     = std::get_if<MtpWork>(&f.request.work.operation);
    if (graphs && (mtp || (forward && forward->kind == ForwardKind::Ordinary))) {
        const auto& geometry = mtp ? mtp->speculative : forward->speculative;
        f.graph_route        = resources.plan.decode_routes->select_route(
            f.physical, geometry ? SpeculativeBackend::Mtp : SpeculativeBackend::None,
            geometry ? geometry->mtp_steps : 0);
    }
    resources.select_batch(f.physical.rows);
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
    if (forward && forward->kind == ForwardKind::Prefill && !forward->visual &&
        !forward->visual_forward) {
        auto& packed = f.packed.emplace();
        packed.last_columns.reserve(f.rows.size());
        for (std::uint32_t b = 0; b < f.rows.size(); ++b) {
            const auto& row = f.rows[b];
            f.tokens += row.ids.size();
            if (!row.ids.empty()) packed.segments.push_back(b);
            if (row.sample) packed.sampled_members.push_back(b);
            packed.last_columns.push_back(row.ids.empty() ? -1 : dimension(f.tokens) - 1);
        }
    }
    resources.compute_actions->submit(
        local_actions(f.request.work.actions, parameters.profile->replica_rank()));
    advance_compute();
}

void RankProgramImpl::advance_compute() {
    if (!compute) return;
    auto& f = *compute;
    for (;;) {
        switch (f.phase) {
        case ComputePhase::Prepare:
#if NINFER_ENABLE_DISTRIBUTED
            if ((pipeline_input && !pipeline_input->available()) ||
                (pipeline_output && !pipeline_output->available()))
                return;
#endif
            if (std::holds_alternative<MtpWork>(f.request.work.operation))
                prepare_mtp();
            else
                prepare_forward();
            if (f.phase == ComputePhase::Prepare) return;
            break;
        case ComputePhase::Prefix:
            if (f.graph ? f.graph->advance_until(*f.graph->hook_layer()) != TargetProgress::Complete
                        : !advance_layers(parameters.profile->config.text.ple.layer))
                return;
            f.phase = ComputePhase::AwaitPle;
            break;
        case ComputePhase::AwaitPle: {
            request_decision();
            if (!host_ple) {
                ple_ready.wait(cuda.stream());
                f.run->consume_ple(device_embedding(f.tokens, f.sequence ? 1 : f.physical.rows));
                f.phase = ComputePhase::Suffix;
                break;
            }
            const auto embedding = host_ple->try_consume(*f.ple);
            if (!embedding) return;
            request_decision();
            if (f.graph) {
                std::optional<nvtx::ScopedRange> range;
                if (f.verify) range.emplace(nvtx::Name::DecodeMtpTarget, nvtx::Category::Mtp);
                if (f.graph->hook_layer()) {
                    f.graph->launch_hook(f.ple->slot);
                } else {
                    f.target_replay.slot = f.ple->slot;
                    (void)graphs->start_target(*f.graph_route, f.target_replay);
                }
            } else {
                embedding->ready.wait(cuda.stream());
                f.run->consume_ple(embedding->values);
                embedding->consumed.record(cuda.stream());
            }
            host_ple->consumer_submitted(*f.ple);
            if (const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
                forward && forward->prefetch) {
                const auto& next = *forward->prefetch;
                const ple::InputRow input{{next.member.sequence.value,
                                           next.member.sequence.generation, next.frontier,
                                           next.raw_tail},
                                          next.ids};
                host_ple->prefetch(next.ids.size(), {&input, 1});
            }
            f.phase =
                f.graph && !f.graph->hook_layer() ? ComputePhase::AfterMain : ComputePhase::Suffix;
            break;
        }
        case ComputePhase::Suffix:
            if (f.graph) {
                if (f.graph->poll() != TargetProgress::Complete) return;
            } else {
                if (!advance_layers(parameters.text.layers.size())) return;
                Tensor logits;
                const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
                if (parameters.profile->output_stage() &&
                    (f.verify || (forward && forward->sample == SampleKind::NextToken)))
                    logits = packed_tensor(*resources.buffers().logits,
                                           {dimension(parameters.profile->local.vocabulary),
                                            static_cast<int>(f.tokens),
                                            static_cast<int>(f.sequence ? 1 : f.physical.rows)});
                finish_text(*f.run, {f.sample, f.conditioning}, logits);
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

            gpu_timer.observe_result(f.gpu);
            events.emplace_back(
                runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(f.result)});
            if (resources.compute_actions->occupied()) resources.compute_actions->finish();
            f.phase = ComputePhase::PendingCommit;
            return;
        case ComputePhase::PendingCommit:
            return;
        case ComputePhase::AwaitExpertService:
            if (!compute_done.ready()) return;
            gpu_timer.observe_result(f.gpu);
            events.emplace_back(runtime::ResultReady<ExecutionTypes>{
                f.request.ticket, moe::ExpertServiceResult{f.request.work.exchange->id}});
            f.phase = ComputePhase::PendingCommit;
            return;
        case ComputePhase::Retiring:
            if (!compute_done.ready() || !consumers_retired(f)) return;
            if (resources.compute_actions->occupied()) resources.compute_actions->finish();
            gpu_timer.observe_retirement(f.gpu);
            publish_retirement(std::move(compute));
            return;
        }
    }
}

void RankProgramImpl::finish_result() {
    auto& f      = *compute;
    auto& result = f.result;
    result.rows.resize(f.rows.size());
    const auto& buffers      = resources.buffers();
    const auto* ids          = static_cast<const TokenId*>(resources.egress().data);
    const auto* forward      = std::get_if<ForwardWork>(&f.request.work.operation);
    const bool sampled       = parameters.profile->output_stage() &&
                         (f.verify || (forward && forward->sample != SampleKind::None));
    const auto probabilities = [&](const TensorRegion& region) {
        return static_cast<const float*>(
            resources.egress()
                .subspan(region.region.offset - buffers.sampled_ids->region.offset)
                .data);
    };
    const auto* top1 =
        forward && forward->sample == SampleKind::NextToken && buffers.top1_probabilities
            ? probabilities(*buffers.top1_probabilities)
            : nullptr;
    const auto* confidence     = f.verify ? probabilities(*buffers.draft_confidences) : nullptr;
    const std::int32_t* counts = nullptr;
    if (f.verify) {
        counts = reinterpret_cast<const std::int32_t*>(
            resources.egress()
                .subspan(buffers.output_counts->region.offset - buffers.sampled_ids->region.offset)
                .data);
    }
    for (std::size_t b = 0; b < f.rows.size(); ++b) {
        auto& row     = result.rows[b];
        row.processed = f.verify ? 0 : f.rows[b].ids.size();
        row.drafts    = f.verify ? f.verify->extents[b] : 0;
        if (top1) row.target_top1_probability = top1[b];
        for (std::uint32_t step = 0; step < row.drafts; ++step)
            row.draft_confidences.push_back(confidence[step * f.physical.rows + b]);
        if (sampled && !f.packed) {
            const auto count = counts ? counts[b] : 1;
            if (count < 1 || static_cast<std::uint32_t>(count) > row.drafts + 1)
                throw std::logic_error("Qwen4Exp generated prefix exceeds its licensed extent");
            const auto stride = f.verify ? f.verify->width : 1;
            row.tokens.assign(ids + stride * b, ids + stride * b + count);
            row.accepted_drafts = count - 1;
            row.dead_positions  = f.dead_positions[b];
        }
        if (auto ns = collect_vision_timing(binding(f.rows[b].member.residency)))
            f.gpu.add_vision(*ns);
    }
    if (f.packed && sampled) {
        const auto& members = f.packed->sampled_members;
        for (std::size_t i = 0; i < members.size(); ++i) {
            auto& row          = result.rows[members[i]];
            row.tokens         = {ids[i]};
            row.dead_positions = f.dead_positions[members[i]];
        }
    }
    if (f.timer_started)
        f.gpu.add_prefill(
            static_cast<std::uint64_t>(double(prefill_timer.elapsed_ms()) * 1.0e6 + 0.5));

    gpu_timer.observe_result(f.gpu);
    if (ExecutionTypes::describe_work(f.request.work).kind == ExecutionKind::Decode)
        gpu_timer.observe_decode(f.gpu, f.verify.has_value());
    events.emplace_back(runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(result)});
    if (resources.compute_actions->occupied()) resources.compute_actions->finish();
    f.phase = ComputePhase::PendingCommit;
}

void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& commit) {
    prepare_commit(ticket, commit);
    auto& slot           = retained.at(ticket.frame.index);
    auto& frame          = slot ? *slot : *compute;
    frame.release_vision = std::ranges::any_of(
        commit.rows, [](const auto& row) { return row.cancelled || row.failed || row.terminal; });
    // The result already fences this ticket's forward work. A width-one pipeline commit
    // with no device mutations owns neither the shared scratch nor another ticket's fence.
    if (retained.size() > 1 &&
        local_actions(commit.actions, parameters.profile->replica_rank()).empty() &&
        !pipeline_counts_needed(frame, commit)) {
        if (shape.gpu_timing) frame.gpu.commit_stream_ns = 0;
        frame.retirement = command;
        frame.phase      = ComputePhase::Retiring;
        if (!slot) slot = std::move(compute);
        return;
    }
    if (!compute || compute->request.ticket != ticket) {
        auto& f           = *retained.at(ticket.frame.index);
        f.deferred_commit = std::move(commit);
        f.retirement      = command;
        return;
    }
    apply_commit(ticket, command, std::move(commit));
}

void RankProgramImpl::apply_commit(runtime::ExecutionTicket ticket,
                                   runtime::CommandSequence command, Commit&& commit) {
    auto& f = *compute;
    gpu_timer.resolve();
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (f.verify || (parameters.mtp && forward && forward->kind == ForwardKind::Ordinary))
        commit_mtp(commit);
    else if (parameters.profile->stage.size > 1 && parameters.profile->output_stage() && forward &&
             forward->sample != SampleKind::None)
        update_pipeline_counts(commit);
    else if (const auto* work = std::get_if<ForwardWork>(&f.request.work.operation);
             parameters.profile->output_stage() && work && work->sample != SampleKind::None) {
        const auto& io = resources.buffers();
        const int batch =
            f.packed ? dimension(f.packed->sampled_members.size()) : dimension(commit.rows.size());
        auto* lengths = static_cast<std::int32_t*>(host_tensor(io.batch.count_columns).data);
        bool any      = false;
        for (int b = 0; b < batch; ++b) {
            const auto member    = f.packed ? f.packed->sampled_members[b] : b;
            const auto& sampling = f.rows[member].sampling;
            lengths[b]           = sampling.presence_penalty || sampling.frequency_penalty
                                       ? commit.rows[member].accepted
                                       : 0;
            any |= lengths[b] != 0;
        }
        if (any) {
            auto columns = device_tensor(io.batch.count_columns).slice(0, 0, batch);
            copy_bytes_async({columns.data, columns.bytes(), cuda.device()},
                             {lengths, columns.bytes(), Device::cpu(), true}, cuda);
            auto counts = device_tensor(*io.token_counts);
            ops::increment_token_counts_batch(
                packed_tensor(*io.sampled_ids, {1, batch}), columns,
                device_tensor(io.batch.count_slots).slice(0, 0, batch), counts, cuda);
        }
    }
    if (!std::holds_alternative<moe::ExpertServiceWork>(f.request.work.operation))
        resources.compute_actions->submit(
            local_actions(commit.actions, parameters.profile->replica_rank()));
    gpu_timer.record_retirement();
    compute_done.record(cuda.stream());
    f.retirement = command;
    f.phase      = ComputePhase::Retiring;
}

namespace {
bool updates_count(const ComputeFrame& frame, const Commit& commit, std::size_t i) {
    const auto& sampling = frame.rows[i].sampling;
    return commit.rows[i].accepted && (sampling.presence_penalty || sampling.frequency_penalty) &&
           !frame.result.rows[i].tokens.empty();
}
} // namespace

bool RankProgramImpl::pipeline_counts_needed(const ComputeFrame& frame,
                                             const Commit& commit) const {
    if (!parameters.profile->output_stage()) return false;
    const auto* forward = std::get_if<ForwardWork>(&frame.request.work.operation);
    if (!forward || forward->sample == SampleKind::None) return false;
    for (std::size_t i = 0; i < commit.rows.size(); ++i)
        if (updates_count(frame, commit, i)) return true;
    return false;
}

void RankProgramImpl::update_pipeline_counts(const Commit& commit) {
    const auto& f = *compute;
    resources.select_batch(f.physical.rows);
    const auto& io = resources.buffers();
    auto host      = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host_tensor(r).data);
    };
    auto* ids     = host(io.anchors);
    auto* slots   = host(io.batch.count_slots);
    auto* lengths = host(io.batch.count_columns);
    int rows      = 0;
    for (std::size_t i = 0; i < commit.rows.size(); ++i) {
        if (!updates_count(f, commit, i)) continue;
        ids[rows]       = f.result.rows.at(i).tokens.at(0);
        slots[rows]     = commit.rows[i].member.residency.lane.value;
        lengths[rows++] = 1;
    }
    if (!rows) return;
    const auto upload = [&](const TensorRegion& region) {
        auto dst = device_tensor(region).slice(0, 0, rows);
        copy_bytes_async({dst.data, dst.bytes(), cuda.device()},
                         {host(region), dst.bytes(), Device::cpu(), true}, cuda);
        return dst;
    };
    auto tokens = upload(io.anchors).view({1, rows});
    auto sizes = upload(io.batch.count_columns), lanes = upload(io.batch.count_slots);
    auto counts = device_tensor(*io.token_counts);
    ops::increment_token_counts_batch(tokens, sizes, lanes, counts, cuda);
}

void RankProgramImpl::discard(runtime::ExecutionTicket ticket, runtime::CommandSequence command) {
    if (!can_discard(ticket)) throw std::logic_error("Qwen4Exp discard has no active execution");
    if (!compute || compute->request.ticket != ticket) {
        auto& f            = *retained.at(ticket.frame.index);
        f.deferred_discard = true;
        f.retirement       = command;
        return;
    }
    apply_discard(ticket, command);
}

void RankProgramImpl::apply_discard(runtime::ExecutionTicket ticket,
                                    runtime::CommandSequence command) {
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
    auto& f = *compute;
    gpu_timer.resolve();
    f.release_vision = true;
    f.retirement     = command;
    if (host_ple) host_ple->cancel(ticket);
    if (f.graph && f.phase != ComputePhase::PendingCommit) f.graph->cancel();
    if (resources.experts) resources.experts->cancel();
    f.run.reset();
    gpu_timer.record_retirement();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::Retiring;
}

bool RankProgramImpl::consumers_retired(const ComputeFrame& frame) {
#if NINFER_ENABLE_DISTRIBUTED
    if (frame.sent && !pipeline_output->retired(*frame.sent)) return false;
#endif
    return !host_ple ||
           std::holds_alternative<moe::ExpertServiceWork>(frame.request.work.operation) ||
           host_ple->retired(frame.request.ticket);
}

void RankProgramImpl::publish_retirement(std::unique_ptr<ComputeFrame> finished) {
    const auto ticket  = finished->request.ticket;
    const auto command = finished->retirement;
    const bool service =
        std::holds_alternative<moe::ExpertServiceWork>(finished->request.work.operation);
    if (!service && resources.vision) {
        const auto* active  = resources.vision->current();
        const auto* forward = std::get_if<ForwardWork>(&finished->request.work.operation);
        if (active && forward && forward->visual && forward->visual->window.id == active->id &&
            (finished->release_vision ||
             forward->rows.front().main_frontier + forward->rows.front().ids.size() >=
                 active->plan->consume_end))
            resources.vision->release(active->id);
    }
    auto observation = retirement_observation();
    observation.gpu  = finished->gpu;
    const auto work  = ExecutionTypes::describe_work(finished->request.work);
    if (finished->attention.forwards) observation.attention = {work.kind, finished->attention};
    observation.path = service && finished->request.work.exchange->graph ? ExecutionPath::Graph
                       : !finished->graph                                ? ExecutionPath::Eager
                       : resources.experts                               ? ExecutionPath::Mixed
                                                                         : ExecutionPath::Graph;
    events.emplace_back(runtime::DeviceRetired{ticket, command, observation});
}

std::optional<runtime::ExecutionEvent<ExecutionTypes>> RankProgramImpl::poll() {
    if (host_ple) host_ple->progress();
    advance_context();
    admit_compute();
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
        if (host_ple) attempt([&] { host_ple->cancel(compute->request.ticket); });
        if (compute->graph) attempt([&] { compute->graph->cancel(); });
    }
    if (resources.experts) attempt([&] { resources.experts->cancel(); });
    if (host_ple) attempt([&] { host_ple->drain(); });
    attempt([&] { resources.synchronize(); });
#if NINFER_ENABLE_DISTRIBUTED
    if (pipeline_input) attempt([&] { pipeline_input->drain(); });
    if (pipeline_output) attempt([&] { pipeline_output->drain(); });
#endif
    compute.reset();
    waiting.clear();
    for (auto& f : retained) f.reset();
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
