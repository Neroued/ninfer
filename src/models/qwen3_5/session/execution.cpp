#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <stdexcept>
#include <type_traits>

namespace ninfer::models::qwen3_5::detail {
std::optional<runtime::ExecutionTicket>
SessionImpl::submit_execution(Execution&& pending, runtime::ExecutionReservation reservation,
                              Work&& work) {
    if (execution) throw std::logic_error("compute execution is already pending");
    execution.emplace(std::move(pending));
    try {
        plan_work_batch(work, *plan.batch_execution);
        execution->preparation->prepare_actions();
        work.context      = execution->preparation->actions();
        const auto ticket = executor().submit(std::move(reservation), std::move(work));
        execution->ticket = ticket;
        for (auto h : execution->members) sequences[lane(h)].computing = true;
        if (execution->vision_id) vision_window.submitted(execution->vision_id, ticket);
        execution->preparation->mark_submitted();
        return ticket;
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

runtime::PrefillCandidateInfo SessionImpl::prefill_candidate(SequenceHandle handle) const {
    const auto l        = lane(handle);
    const auto& state   = sequences[l];
    const auto& request = requests[l];
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill)
        throw std::logic_error("prefill candidate is outside prefill");
    const auto begin  = state.continuation.metadata.execution_frontier;
    const auto prompt = request.prefill->prompt_tokens;
    const auto end    = window_end(state, request, begin, prompt, vision_window);
    const bool capture =
        request.capture_reservation && request.capture_reservation->frontier == begin;
    const auto* active_window = vision_window.current();
    const bool carries_visual = active_window && active_window->owner.sequence == state.id &&
                                active_window->owner.residency == state.residency;
    const bool exclusive =
        request.base->vision && begin < end &&
        (carries_visual ||
         std::ranges::any_of(request.base->vision->units, [&](const models::vision::Unit& unit) {
             return unit.token_begin < end && begin < unit.end();
         }));
    return {.remaining_tokens  = prompt - begin,
            .unit_max_tokens   = end - begin,
            .exclusive         = exclusive,
            .zero_token_sample = begin == prompt && !capture};
}

std::optional<runtime::ExecutionTicket> SessionImpl::start_prefill(SequenceHandle handle,
                                                                   bool constrained) {
    if (execution) return {};
    const runtime::PrefillGrant grant{prefill_candidate(handle).unit_max_tokens, constrained};
    return start_prefill({&handle, 1}, {&grant, 1});
}

std::optional<runtime::ExecutionTicket>
SessionImpl::start_prefill(std::span<const SequenceHandle> members,
                           std::span<const runtime::PrefillGrant> grants) {
    if (execution) return {};
    if (members.empty() || members.size() > max_concurrency || grants.size() != members.size())
        throw std::logic_error("invalid prefill membership");
    std::vector<std::uint32_t> ends;
    ends.reserve(members.size());
    const auto mark         = next_lane_mark();
    std::uint32_t remaining = prefill_chunk;
    for (std::size_t b = 0; b < members.size(); ++b) {
        const auto l        = lane(members[b]);
        const auto& request = requests[l];
        const auto info     = prefill_candidate(members[b]);
        if (!request.permit || request.permit->kind != ExecutionUnitKind::Prefill ||
            request.capture_pending || lane_marks[l] == mark ||
            (info.exclusive && members.size() > 1))
            throw std::logic_error("prefill has no distinct executable window");
        lane_marks[l]     = mark;
        const auto& state = sequences[l];
        const auto begin  = state.continuation.metadata.execution_frontier;
        const auto limit =
            begin + std::min(grants[b].max_tokens, request.prefill->prompt_tokens - begin);
        const auto end = window_end(state, request, begin,
                                    std::min(limit, request.permit->main_frontier), vision_window);
        if (end < begin || (end == begin && !info.zero_token_sample))
            throw std::logic_error("prefill grant has no executable window");
        if (end - begin > remaining)
            throw std::logic_error("prefill grants exceed the token budget");
        remaining -= end - begin;
        ends.push_back(end);
    }
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    try {
        Execution pending{.kind = OperationKind::Prefill};
        pending.members.assign(members.begin(), members.end());
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        pending.bases.reserve(members.size());
        pending.columns.reserve(members.size());
        pending.sampled.reserve(members.size());
        ForwardWork forward;
        forward.purpose = ForwardPurpose::Prefill;
        forward.rows.reserve(members.size());
        forward.inputs.reserve(members.size());
        forward.sampling.reserve(members.size());
        for (std::size_t b = 0; b < members.size(); ++b) {
            const auto l  = lane(members[b]);
            auto& state   = sequences[l];
            auto& request = requests[l];
            prepare_capture_boundary(l);
            const auto begin = state.continuation.metadata.execution_frontier;
            // A newly reserved capture can shorten the validated window.
            const auto end = window_end(state, request, begin, ends[b], vision_window);
            const bool capture =
                request.capture_reservation && request.capture_reservation->frontier == end;
            const bool sample = end == request.prefill->prompt_tokens && !capture;
            pending.bases.push_back(state.continuation);
            pending.columns.push_back(end - begin);
            pending.sampled.push_back(sample);
            ensure_sequence_kv_mapped(state, end,
                                      speculative_backend == SpeculativeBackend::Mtp
                                          ? (end ? end - 1 : 0)
                                      : has_backend_pool() ? end
                                                           : 0,
                                      *pending.preparation);
            forward.rows.push_back(execution_row(state));
            forward.inputs.push_back(input_window(state, begin, end));
            forward.inputs.back().visual = prepare_visual(state, begin, end);
            if (forward.inputs.back().visual)
                pending.vision_id = forward.inputs.back().visual->window.id;
            if (sample) forward.sample = SamplePurpose::FirstToken;
            forward.sampling.push_back(
                {request.sampling_host, sample && grants[b].constrained, sample});
        }
        if (forward.sample == SamplePurpose::None) forward.sampling.clear();
        return submit_execution(std::move(pending), std::move(*reservation),
                                Work{.operation = std::move(forward)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket> SessionImpl::start_replay(SequenceHandle handle) {
    if (execution) return {};
    const auto l  = lane(handle);
    auto& state   = sequences[l];
    auto& request = requests[l];
    if (request.lifecycle != Lifecycle::Replaying || request.capture_pending)
        throw std::logic_error("sequence has no executable Replay window");
    require_unit(l, ExecutionUnitKind::Replay);
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    try {
        prepare_capture_boundary(l);
        const auto begin = state.continuation.metadata.execution_frontier;
        const auto end   = window_end(state, request, begin,
                                      std::min(request.permit->main_frontier, request.replay_target),
                                      vision_window);
        Execution pending{.kind = OperationKind::Replay};
        pending.members     = {handle};
        pending.bases       = {state.continuation};
        pending.columns     = {end - begin};
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        ensure_sequence_kv_mapped(state, end,
                                  speculative_backend == SpeculativeBackend::Mtp
                                      ? (end ? end - 1 : 0)
                                  : has_backend_pool() ? end
                                                       : 0,
                                  *pending.preparation);
        ForwardWork forward;
        forward.purpose = ForwardPurpose::Replay;
        forward.rows.push_back(execution_row(state));
        forward.inputs.push_back(input_window(state, begin, end));
        forward.inputs.back().visual = prepare_visual(state, begin, end);
        if (forward.inputs.back().visual)
            pending.vision_id = forward.inputs.back().visual->window.id;
        return submit_execution(std::move(pending), std::move(*reservation),
                                Work{.operation = std::move(forward)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
SessionImpl::start_control(std::span<const SequenceHandle> members, std::span<const TokenId> tokens,
                           std::uint32_t stride) {
    if (execution) return {};
    if (members.empty() || members.size() > max_concurrency || !stride ||
        stride > plan.profile->control_token_limit || tokens.size() != members.size() * stride)
        throw std::invalid_argument("forced control shape is invalid");
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    try {
        Execution pending{.kind = OperationKind::Control};
        pending.members.assign(members.begin(), members.end());
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        ForwardWork forward;
        forward.purpose = ForwardPurpose::Control;
        forward.rows.reserve(members.size());
        forward.inputs.reserve(members.size());
        forward.sampling.reserve(members.size());
        forward.control_last_tokens.reserve(members.size());
        pending.bases.reserve(members.size());
        pending.columns.reserve(members.size());
        pending.inputs.reserve(members.size());
        const auto mark = next_lane_mark();
        for (std::size_t b = 0; b < members.size(); ++b) {
            const auto l = lane(members[b]);
            if (lane_marks[l] == mark) throw std::logic_error("duplicate sequence in control");
            lane_marks[l] = mark;
            auto& state   = sequences[l];
            auto& request = requests[l];
            const auto f  = state.continuation.metadata.execution_frontier;
            if (request.lifecycle != Lifecycle::Active || state.ledger.size() != f + 1)
                throw std::logic_error("forced control sequence is not ready");
            require_unit(l, ExecutionUnitKind::Control, stride);
            const auto forced = tokens.subspan(b * stride, stride);
            for (const auto id : forced)
                if (id < 0 || static_cast<std::uint32_t>(id) >= plan.profile->public_tokens)
                    throw std::invalid_argument("forced token is outside the public vocabulary");
            pending.bases.push_back(state.continuation);
            pending.columns.push_back(stride);
            pending.inputs.emplace_back(forced.begin(), forced.end());
            ensure_sequence_kv_mapped(state, f + stride,
                                      speculative_backend == SpeculativeBackend::Mtp
                                          ? f + stride - 1
                                      : has_backend_pool() ? f + stride
                                                           : 0,
                                      *pending.preparation);
            auto input = input_window(state, f, f + 1);
            input.tokens.insert(input.tokens.end(), forced.begin(), forced.end() - 1);
            input.rope_positions.resize(3ULL * stride);
            for (int axis = 0; axis < 3; ++axis)
                for (std::uint32_t j = 0; j < stride; ++j)
                    input.rope_positions[axis * stride + j] =
                        static_cast<std::int32_t>(f + j) + state.continuation.metadata.rope_delta;
            forward.rows.push_back(execution_row(state));
            forward.inputs.push_back(std::move(input));
            forward.control_last_tokens.push_back(forced.back());
            forward.sampling.push_back({request.sampling_host, false});
        }
        return submit_execution(std::move(pending), std::move(*reservation),
                                Work{.operation = std::move(forward)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
SessionImpl::start_normalize(SequenceHandle handle, runtime::ExecutionReservation reservation) {
    if (execution) return {};
    const auto l = lane(handle);
    auto& state  = sequences[l];
    require_unit(l, ExecutionUnitKind::Normalize);
    try {
        Execution pending{.kind = OperationKind::Normalize};
        pending.members     = {handle};
        pending.bases       = {state.continuation};
        pending.columns     = {0};
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        ensure_sequence_kv_mapped(
            state, state.continuation.metadata.text_kv_valid,
            has_backend_pool() ? state.continuation.metadata.execution_frontier : 0,
            *pending.preparation);
        NormalizeWork work{{execution_row(state)}};
        return submit_execution(std::move(pending), std::move(reservation),
                                Work{.operation = std::move(work)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

void SessionImpl::supply(runtime::ExecutionTicket ticket, runtime::DecisionInput&& input) {
    if (!execution || execution->ticket != ticket || execution->result_ready)
        throw std::logic_error("decision input does not match a pending execution");
    executor().supply(ticket, std::move(input));
}

std::optional<SessionEvent> SessionImpl::poll() {
    try {
        for (;;) {
            submit_releases();
            if (!events.empty()) {
                auto event = std::move(events.front());
                events.pop_front();
                if (scoring && std::holds_alternative<ContextProgress>(event.result)) {
                    const auto& progress = std::get<ContextProgress>(event.result);
                    if (!progress.published || !progress.sequence)
                        throw std::logic_error("scoring context did not bind");
                    scoring->sequence = *progress.sequence;
                    advance_score();
                    continue;
                }
                return event;
            }
            auto event = executor().poll();
            if (!event) return {};
            if (const auto* failed = std::get_if<runtime::ExecutionFailed>(&*event))
                throw std::runtime_error(failed->message);
            std::visit(
                [&](auto&& value) {
                    using T = std::remove_cvref_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, runtime::ResultReady<ExecutionTypes>>) {
                        receive_result(std::move(value));
                    } else if constexpr (std::is_same_v<T, runtime::DeviceRetired>) {
                        retire(value);
                    } else if constexpr (std::is_same_v<T, runtime::DecisionNeeded>) {
                        if (!execution || execution->ticket != value.ticket)
                            throw std::logic_error("decision request has no owning execution");
                        events.push_back(
                            {value.ticket, execution->members, std::move(value.request)});
                    } else if constexpr (std::is_same_v<T, runtime::DecisionConsumed>) {
                        if (!execution || execution->ticket != value.ticket)
                            throw std::logic_error("decision consumption has no owning execution");
                        events.push_back({value.ticket, execution->members, std::move(value)});
                    }
                },
                std::move(*event));
        }
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
