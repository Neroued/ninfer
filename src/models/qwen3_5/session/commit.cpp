#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

PendingBatch SessionImpl::generated_pending(const GeneratedResult& generated) {
    auto& current    = *execution;
    const auto batch = current.members.size();
    if (!generated.row_stride || generated.tokens.size() != batch * generated.row_stride ||
        generated.counts.size() != batch || generated.dead_positions.size() != batch)
        throw std::logic_error("generated result has an invalid shape");
    if (current.kind == OperationKind::Decode &&
        (generated.draft_extents.size() != batch || generated.accepted_drafts.size() != batch))
        throw std::logic_error("decode result omitted its proposal extents");
    PendingBatch out;
    out.ticket_    = current.ticket;
    out.members_   = current.members;
    out.tokens_    = generated.tokens;
    out.stride_    = generated.row_stride;
    out.execution_ = current.observed_result;
    for (std::size_t b = 0; b < batch; ++b) {
        const auto count = generated.counts[b];
        if (!count || count > generated.row_stride || count > 16)
            throw std::logic_error("generated result has an invalid licensed prefix");
        out.counts_.push_back(static_cast<std::int32_t>(count));
        out.failed_.push_back((generated.dead_positions[b] & ((1U << count) - 1)) != 0);
        auto& state       = sequences[lane(current.members[b])];
        auto& request     = requests[state.lane];
        request.lifecycle = Lifecycle::Pending;
    }
    return out;
}

namespace {
ContinuationMetadata forward_metadata(ContinuationMetadata value, std::uint32_t count,
                                      SpeculativeBackend backend, bool backend_pool) {
    if (!count) return value;
    value.execution_frontier += count;
    value.text_kv_valid           = value.execution_frontier;
    value.tail_hidden_valid       = true;
    value.dflash_context_frontier = value.execution_frontier;
    value.backend_kv_valid = backend == SpeculativeBackend::Mtp ? value.execution_frontier - 1
                             : backend_pool                     ? value.execution_frontier
                                                                : 0;
    return value;
}

void prepare_identity(SequenceState& state, std::size_t count, std::optional<std::uint32_t> split) {
    if (state.prefix_identity.size() != state.ledger.size() ||
        state.prefix_digests.size() != state.ledger.size())
        throw std::logic_error("generated identity does not match its ledger");
    state.ledger.reserve(state.ledger.size() + count);
    state.prefix_identity.prepare_generated(count, state.continuation.metadata.rope_delta, split);
    state.prefix_digests.prepare_generated(count, state.continuation.metadata.rope_delta, split);
}

void append_identity(SequenceState& state, std::span<const TokenId> tokens,
                     std::optional<std::uint32_t> split) noexcept {
    state.ledger.insert(state.ledger.end(), tokens.begin(), tokens.end());
    state.prefix_identity.append_generated(tokens.size(), state.continuation.metadata.rope_delta,
                                           split);
    state.prefix_digests.append_generated(tokens, state.continuation.metadata.rope_delta, split);
}
} // namespace

void SessionImpl::receive_result(runtime::ResultReady<ExecutionTypes>&& event) {
    if (auto* context = std::get_if<ContextResult>(&event.result.output)) {
        if (release_ticket && *release_ticket == event.ticket) {
            if (release_result) throw std::logic_error("duplicate residency release result");
            release_result = true;
            return;
        }
        receive_context_result(event.ticket, std::move(*context));
        return;
    }
    if (!execution || execution->ticket != event.ticket || execution->result_ready)
        throw std::logic_error("execution result does not match its transaction");
    auto& current           = *execution;
    current.result_ready    = true;
    current.observed_result = event.execution;

    if (auto* generated = std::get_if<GeneratedResult>(&event.result.output)) {
        if (current.kind != OperationKind::Decode)
            throw std::logic_error("decode result has the wrong execution kind");
        current.generated.emplace(std::move(*generated));
        auto pending = generated_pending(*current.generated);
        events.push_back({current.ticket, current.members, std::move(pending)});
        return;
    }
    if (auto* forward = std::get_if<ForwardResult>(&event.result.output)) {

        if (forward->processed_tokens != current.columns)
            throw std::logic_error("forward result changed its reserved input window");
        for (std::size_t b = 0; b < current.members.size(); ++b)
            current.completed.push_back(forward_metadata(current.bases[b].metadata,
                                                         current.columns[b], speculative_backend,
                                                         has_backend_pool()));
        const auto& request = requests[lane(current.members[0])];
        const auto end      = current.completed[0].execution_frontier;
        const bool capture =
            request.capture_reservation && request.capture_reservation->frontier == end;
        if (current.kind == OperationKind::Prefill) {
            PrefillProgress progress;
            progress.summary = {request.base->summary.prompt_tokens, request.prefill->base,
                                request.prefill->reuse};
            progress.processed_prompt_tokens = current.columns[0];
            progress.capture_ready           = capture && !forward->generated;
            progress.complete                = end == request.base->summary.prompt_tokens;

            if (forward->generated) {
                if (!progress.complete)
                    throw std::logic_error("incomplete prefill produced a token");
                current.generated.emplace(std::move(*forward->generated));
                progress.pending.emplace(generated_pending(*current.generated));
                events.push_back({current.ticket, current.members, std::move(progress)});
                return;
            }
            if (progress.complete)
                throw std::logic_error("completed prefill omitted its first token");
            current.completion.emplace(
                SessionEvent{current.ticket, current.members, std::move(progress)});
        } else if (current.kind == OperationKind::Replay) {
            current.completion.emplace(
                SessionEvent{current.ticket, current.members,
                             ReplayProgress{capture, current.columns[0],
                                            end == request.replay_target && !capture}});
        } else if (current.kind == OperationKind::Control) {
            current.completion.emplace(
                SessionEvent{current.ticket, current.members, ControlComplete{}});
        } else
            throw std::logic_error("unexpected forward result");
        commit_completed_unit();
        return;
    }
    if (std::holds_alternative<NormalizeResult>(event.result.output)) {
        if (current.kind != OperationKind::Normalize)
            throw std::logic_error("normalize result has the wrong execution kind");
        auto metadata                    = current.bases[0].metadata;
        metadata.dflash_context_frontier = metadata.execution_frontier;
        metadata.backend_kv_valid        = has_backend_pool() ? metadata.execution_frontier : 0;
        current.completed.push_back(metadata);
        commit_completed_unit();
        return;
    }
    if (auto* score = std::get_if<ScoreResult>(&event.result.output)) {
        if (!scoring || current.kind != OperationKind::Score ||
            score->processed_tokens != current.columns[0])
            throw std::logic_error("score result has the wrong execution window");
        current.completed.push_back(forward_metadata(
            current.bases[0].metadata, score->processed_tokens, SpeculativeBackend::None, false));
        if (score->logprobs.size() > scoring->logprobs.capacity() - scoring->logprobs.size())
            throw std::logic_error("score result exceeds its prepared target capacity");
        commit_completed_unit();
        scoring->cursor += score->processed_tokens;
        scoring->logprobs.insert(scoring->logprobs.end(), score->logprobs.begin(),
                                 score->logprobs.end());

        return;
    }
    throw std::logic_error("unknown native result");
}

void SessionImpl::commit_completed_unit() {
    auto& current      = *execution;
    current.settlement = std::make_unique<context::ContextTransaction>(directory);
    std::array<bool, kMaximumConcurrency> recovery{};
    Commit physical;
    physical.rows.reserve(current.members.size());
    for (std::size_t b = 0; b < current.members.size(); ++b) {
        const auto l        = lane(current.members[b]);
        auto& state         = sequences[l];
        const auto appended = current.kind == OperationKind::Control ? current.inputs[b].size() : 0;
        if (appended) prepare_identity(state, appended, current.splits[b]);
        recovery[b] = prepare_settlement(l, current.completed[b], requests[l].lifecycle,
                                         state.ledger.size() + appended, *current.settlement);
        physical.rows.push_back({current.columns[b], false, false});
    }
    current.settlement->prepare_actions();
    physical.context = current.settlement->actions();
    (void)executor().commit(current.ticket, std::move(physical));
    current.settlement->mark_submitted();
    current.settlement->adopt();
    current.committed = true;
    for (std::size_t b = 0; b < current.members.size(); ++b) {
        const auto l  = lane(current.members[b]);
        auto& state   = sequences[l];
        auto& request = requests[l];
        adopt_unit(l, current.completed[b],
                   current.columns[b] != 0 || current.kind == OperationKind::Normalize,
                   recovery[b]);
        const auto end = current.completed[b].execution_frontier;
        request.capture_pending =
            request.capture_reservation && request.capture_reservation->frontier == end;
        if (current.kind == OperationKind::Prefill) request.prefill->cursor = end;
        if (current.kind == OperationKind::Replay) request.replay_cursor = end;
        if (current.kind == OperationKind::Control)
            append_identity(state, current.inputs[b], current.splits[b]);
    }
}

CommitResult SessionImpl::commit(PendingBatch&& pending,
                                 std::span<const runtime::CommitDecision> decisions) {
    if (!execution || execution->ticket != pending.ticket_ ||
        execution->members != pending.members_ || execution->committed || !execution->generated ||
        decisions.size() != pending.row_count())
        throw std::logic_error("generated commit has no matching transaction");
    auto& current = *execution;
    for (std::size_t b = 0; b < decisions.size(); ++b) {
        const auto& d = decisions[b];
        if ((d.cancelled && d.failed) || (d.failed && !pending.failed_[b]) ||
            ((d.cancelled || d.failed)
                 ? (!d.terminal || d.accepted_tokens != 0)
                 : (!d.accepted_tokens ||
                    d.accepted_tokens > static_cast<std::uint32_t>(pending.counts_[b]) ||
                    (!d.terminal &&
                     d.accepted_tokens != static_cast<std::uint32_t>(pending.counts_[b])) ||
                    pending.failed_[b])) ||
            (d.prefix_execution_split_after &&
             (!*d.prefix_execution_split_after ||
              *d.prefix_execution_split_after > d.accepted_tokens)))
            throw std::logic_error("invalid accepted-prefix decision");
    }
    CommitResult result;
    result.row_count = decisions.size();
    std::array<bool, kMaximumConcurrency> recovery{};
    try {
        current.decisions.assign(decisions.begin(), decisions.end());
        current.settlement = std::make_unique<context::ContextTransaction>(directory);
        if (current.kind == OperationKind::Decode) {
            current.completed.reserve(decisions.size());
            for (std::size_t b = 0; b < decisions.size(); ++b) {
                auto metadata =
                    forward_metadata(current.bases[b].metadata, decisions[b].accepted_tokens,
                                     speculative_backend, has_backend_pool());
                if (is_masked_draft_backend(speculative_backend)) {
                    metadata.dflash_context_frontier =
                        decisions[b].terminal ? metadata.execution_frontier
                                              : current.bases[b].metadata.execution_frontier;
                    metadata.backend_kv_valid =
                        has_backend_pool() ? metadata.dflash_context_frontier : 0;
                }
                current.completed.push_back(metadata);
            }
        }
        Commit physical;
        physical.rows.reserve(decisions.size());
        for (std::size_t b = 0; b < decisions.size(); ++b) {
            const auto& d = decisions[b];
            const auto l  = lane(current.members[b]);
            auto& state   = sequences[l];
            result.rows[b].speculative.attempted_per_position.reserve(
                requests[l].speculative_stats.attempted_per_position.size());
            result.rows[b].speculative.accepted_per_position.reserve(
                requests[l].speculative_stats.accepted_per_position.size());
            physical.rows.push_back({d.accepted_tokens, d.terminal, d.cancelled || d.failed});
            if (d.cancelled || d.failed) continue;
            prepare_identity(state, d.accepted_tokens, d.prefix_execution_split_after);
            recovery[b] = prepare_settlement(
                l, current.completed[b], d.terminal ? Lifecycle::Finishable : Lifecycle::Active,
                state.ledger.size() + d.accepted_tokens, *current.settlement);
        }
        current.settlement->prepare_actions();
        physical.context = current.settlement->actions();
        (void)executor().commit(current.ticket, std::move(physical));
        current.settlement->mark_submitted();
        current.settlement->adopt();
        current.committed = true;
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
    for (std::size_t b = 0; b < decisions.size(); ++b) {
        const auto& d = decisions[b];
        const auto l  = lane(current.members[b]);
        auto& state   = sequences[l];
        auto& request = requests[l];
        if (!d.cancelled && !d.failed) {
            adopt_unit(l, current.completed[b],
                       current.kind == OperationKind::Decode || current.columns[b] != 0,
                       recovery[b]);
            if (current.kind == OperationKind::Prefill)
                request.prefill->cursor = current.completed[b].execution_frontier;
            const auto accepted = std::span<const TokenId>(pending.tokens_)
                                      .subspan(b * pending.stride_, d.accepted_tokens);
            append_identity(state, accepted, d.prefix_execution_split_after);
            request.lifecycle = d.terminal ? Lifecycle::Finishable : Lifecycle::Active;
            request.capture_pending =
                request.capture_reservation &&
                request.capture_reservation->frontier == current.completed[b].execution_frontier;
        }
        if (d.terminal) {
            request.capture_pending = false;
            request.capture_reservation.reset();
            request.prefill.reset();
        } else if (current.kind == OperationKind::Prefill && !request.capture_pending) {
            request.prefill.reset();
        }
        auto& stats = request.speculative_stats;
        if (current.kind == OperationKind::Decode &&
            speculative_backend != SpeculativeBackend::None) {
            const auto drafted  = current.generated->draft_extents.at(b);
            const auto accepted = current.generated->accepted_drafts.at(b);
            stats.committed_draft_tokens += std::min(d.accepted_tokens, accepted);
            if (drafted)
                ++stats.rounds;
            else
                ++stats.fallback_steps;
            stats.drafted_tokens += drafted;
            for (std::uint32_t i = 0; i < drafted; ++i) ++stats.attempted_per_position[i];
            stats.accepted_tokens += accepted;
            for (std::uint32_t j = 0; j < accepted; ++j) ++stats.accepted_per_position[j];
        }
        auto& row               = result.rows[b];
        row.disposition         = d.cancelled  ? runtime::CommitDisposition::CancelledReleased
                                  : d.failed   ? runtime::CommitDisposition::FailedReleased
                                  : d.terminal ? runtime::CommitDisposition::Finishable
                                               : runtime::CommitDisposition::Active;
        row.speculative         = stats;
        result.capture_ready[b] = !d.terminal && request.capture_pending;
    }
    pending.ticket_ = {};
    return result;
}

DiscardResult SessionImpl::discard(PendingBatch&& pending) noexcept {
    if (!execution || execution->ticket != pending.ticket_ ||
        execution->members != pending.members_ || execution->committed)
        return {};
    try {
        executor().discard(execution->ticket);
        execution->discarded = true;
        execution->committed = true;
        pending.ticket_      = {};
        return {runtime::ConsumeStatus::Consumed, execution->members.size()};
    } catch (...) { return {}; }
}

void SessionImpl::retire(runtime::DeviceRetired event) {
    if (release_ticket && *release_ticket == event.ticket) {
        if (!release_result) throw std::logic_error("residency retired before release result");
        release_ticket.reset();
        release_result = false;
        if (after_releases && releases.empty()) {
            events.push_back(std::move(*after_releases));
            after_releases.reset();
        }
        return;
    }
    if (event.ticket.frame.kind == runtime::FrameKind::Context) {
        retire_context(event);
        return;
    }
    if (!execution || execution->ticket != event.ticket || !execution->committed)
        throw std::logic_error("execution retired before its resolution");
    if (execution->vision_id) {
        const auto* window = vision_window.current();
        bool complete      = execution->discarded;
        if (window && valid_sequence(window->owner.sequence)) {
            const auto& state = sequences[lane(window->owner.sequence)];
            complete |=
                state.continuation.metadata.execution_frontier >= window->plan->consume_end &&
                (speculative_backend != SpeculativeBackend::Mtp ||
                 backend_kv_valid(state) >= window->plan->consume_end - 1);
        } else
            complete = true;
        vision_window.consumed(execution->vision_id, complete);
        vision_window.retired(event.ticket);
    }
    auto current = std::move(*execution);
    execution.reset();
    current.preparation->mark_completed();
    current.preparation->publish();
    if (current.settlement) {
        current.settlement->mark_completed();
        current.settlement->publish();
    }
    for (std::size_t b = 0; b < current.members.size(); ++b) {
        if (!valid_sequence(current.members[b])) continue;
        const auto l  = lane(current.members[b]);
        auto& request = requests[l];
        if (current.kind == OperationKind::Replay &&
            request.replay_cursor == request.replay_target && !request.capture_pending) {
            request.lifecycle = request.resume_lifecycle;
            if (request.lifecycle == Lifecycle::Prefilling)
                initialize_captures(request, request.replay_target,
                                    request.base->summary.prompt_tokens);
        }
    }
    if (current.kind != OperationKind::Normalize && current.kind != OperationKind::Score)
        events.push_back({current.ticket, current.members,
                          runtime::ExecutionComplete{event.execution, current.members.size()}});
    for (std::size_t b = 0; b < current.members.size(); ++b) {
        if (!valid_sequence(current.members[b])) continue;
        const auto l  = lane(current.members[b]);
        auto& state   = sequences[l];
        auto& request = requests[l];
        if (current.discarded || (!current.decisions.empty() &&
                                  (current.decisions[b].cancelled || current.decisions[b].failed)))
            clear_lane(state, request);
    }
    if (current.completion) events.push_back(std::move(*current.completion));
    if (current.kind == OperationKind::Normalize && normalizing_pause_)
        continue_pause_after_normalize(*normalizing_pause_);
    if (current.kind == OperationKind::Score) advance_score();
}

} // namespace ninfer::models::qwen3_5::detail
