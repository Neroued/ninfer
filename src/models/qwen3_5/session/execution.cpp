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
        execution->preparation->prepare_actions();
        work.context      = execution->preparation->actions();
        const auto ticket = executor().submit(std::move(reservation), std::move(work));
        execution->ticket = ticket;
        execution->preparation->mark_submitted();
        return ticket;
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket> SessionImpl::start_prefill(SequenceHandle handle,
                                                                   bool constrained) {
    if (execution) return {};
    const auto l  = lane(handle);
    auto& state   = sequences[l];
    auto& request = requests[l];
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill || request.capture_pending)
        throw std::logic_error("sequence has no executable prefill window");
    require_unit(l, ExecutionUnitKind::Prefill);
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    try {
        prepare_capture_boundary(l);
        const auto begin = state.continuation.metadata.execution_frontier;
        const auto end   = window_end(
            l, begin, std::min(request.permit->main_frontier, request.base->summary.prompt_tokens));
        const bool sample = end == request.base->summary.prompt_tokens;
        Execution pending{.kind = OperationKind::Prefill};
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
        forward.purpose = ForwardPurpose::Prefill;
        forward.rows.push_back(execution_row(state));
        forward.inputs.push_back(input_window(state, begin, end));
        forward.sample = sample ? SamplePurpose::FirstToken : SamplePurpose::None;
        forward.sampling.push_back({request.sampling_host, constrained});
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
        const auto end =
            window_end(l, begin, std::min(request.permit->main_frontier, request.replay_target));
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
        return submit_execution(std::move(pending), std::move(*reservation),
                                Work{.operation = std::move(forward)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
SessionImpl::start_decode(std::span<const SequenceHandle> members,
                          std::span<const runtime::RoundBudget> budgets,
                          std::span<const bool> constrained) {
    if (execution) return {};
    if (members.empty() || members.size() > max_concurrency || budgets.size() != members.size() ||
        (!constrained.empty() && constrained.size() != members.size()))
        throw std::invalid_argument("decode membership is invalid");
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    try {
        Execution pending{.kind = OperationKind::Decode};
        pending.members.assign(members.begin(), members.end());
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        DecodeWork decode;
        for (std::size_t b = 0; b < members.size(); ++b) {
            if (std::find(members.begin(), members.begin() + b, members[b]) != members.begin() + b)
                throw std::logic_error("duplicate sequence in decode");
            const auto l  = lane(members[b]);
            auto& state   = sequences[l];
            auto& request = requests[l];
            const auto f  = state.continuation.metadata.execution_frontier;
            if (request.lifecycle != Lifecycle::Active || state.ledger.size() != f + 1 ||
                !budgets[b].generated_tokens_remaining)
                throw std::logic_error("sequence is not decode ready");
            require_unit(l, ExecutionUnitKind::Decode, budgets[b].generated_tokens_remaining);
            const auto k = speculative_backend == SpeculativeBackend::None
                               ? 0U
                               : std::min({draft_window, budgets[b].generated_tokens_remaining - 1,
                                           capacity - f - 1});
            pending.bases.push_back(state.continuation);
            pending.columns.push_back(k + 1);
            ensure_sequence_kv_mapped(state, f + k + 1,
                                      speculative_backend == SpeculativeBackend::Mtp ? f + k
                                      : has_backend_pool()                           ? f + k + 1
                                                                                     : 0,
                                      *pending.preparation);
            decode.rows.push_back(
                {execution_row(state),
                 state.ledger.back(),
                 k,
                 budgets[b].generated_tokens_remaining,
                 {request.sampling_host, !constrained.empty() && constrained[b]}});
        }
        return submit_execution(std::move(pending), std::move(*reservation),
                                Work{.operation = std::move(decode)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket>
SessionImpl::start_control(std::span<const SequenceHandle> members, std::span<const TokenId> tokens,
                           std::uint32_t stride,
                           std::span<const std::optional<std::uint32_t>> splits) {
    if (execution) return {};
    if (members.empty() || members.size() > max_concurrency || !stride ||
        tokens.size() != members.size() * stride || splits.size() != members.size())
        throw std::invalid_argument("forced control shape is invalid");
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
    try {
        Execution pending{.kind = OperationKind::Control};
        pending.members.assign(members.begin(), members.end());
        pending.splits.assign(splits.begin(), splits.end());
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        ForwardWork forward;
        forward.purpose = ForwardPurpose::Control;
        for (std::size_t b = 0; b < members.size(); ++b) {
            const auto l  = lane(members[b]);
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
            if (splits[b] && (!*splits[b] || *splits[b] > stride))
                throw std::invalid_argument("forced control split is outside its span");
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
            forward.committed_tokens.emplace_back(forced.begin(), forced.end());
            forward.sampling.push_back({request.sampling_host, false});
        }
        return submit_execution(std::move(pending), std::move(*reservation),
                                Work{.operation = std::move(forward)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

std::optional<runtime::ExecutionTicket> SessionImpl::start_normalize(SequenceHandle handle) {
    if (execution) return {};
    const auto l = lane(handle);
    auto& state  = sequences[l];
    require_unit(l, ExecutionUnitKind::Normalize);
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return {};
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
        return submit_execution(std::move(pending), std::move(*reservation),
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
            if (pending_maintenance_ && !execution && !has_context_operation()) {
                auto assessment = std::move(*pending_maintenance_);
                pending_maintenance_.reset();
                (void)start_maintenance(assessment);
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
