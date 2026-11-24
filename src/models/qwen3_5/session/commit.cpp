#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

PendingBatch SessionImpl::generated_pending(const GeneratedResult& generated) {
    auto& current    = *execution;
    const auto batch = current.members.size();
    if (!generated.row_stride ||
        (current.kind == OperationKind::Prefill && generated.row_stride != 1) ||
        generated.tokens.size() != batch * generated.row_stride ||
        generated.counts.size() != batch || generated.dead_positions.size() != batch)
        throw std::logic_error("generated result has an invalid shape");
    const auto stride =
        current.round ? current.round->route().mtp_steps + current.round->route().head_columns : 0U;
    if (current.round) {
        const auto& initial = *current.round;
        const auto path     = plan.decode_path;
        if (!generated.executed || generated.executed->verify_drafts.size() != batch)
            throw std::logic_error("decode result omitted its executed route or extents");
        const auto& [route, extents] = *generated.executed;
        if (route.shape != initial.route().shape || route.width > initial.route().width ||
            generated.row_stride != route.width ||
            route.graph_entry.has_value() != initial.route().graph_entry.has_value())
            throw std::logic_error("executed target route exceeds its frozen route");
        if (route.graph_entry) {
            const auto prepared = plan.decode_routes->at(route.shape);
            if (std::ranges::find(prepared, route) == prepared.end())
                throw std::logic_error("executed target used an unavailable graph");
        }
        std::vector<speculative::RoundMember> members;
        std::vector<std::uint32_t> frontiers;
        members.reserve(batch);
        frontiers.reserve(batch);
        for (std::size_t b = 0; b < batch; ++b) {
            const auto limit = initial.verify_drafts()[b];
            if (extents[b] > limit)
                throw std::logic_error("executed target extent exceeds its frozen extent");
            members.push_back({initial.slots()[b].owner, limit, limit});
            frontiers.push_back(current.bases[b].metadata.execution_frontier);
        }
        if (route != initial.route() || !std::ranges::equal(extents, initial.verify_drafts())) {
            auto target = speculative::DecodeRoundPlan::freeze(speculative_backend, draft_window, 1,
                                                               members, route, extents);
            current.observed_result.work.speculative =
                std::make_shared<const SpeculativeExecutionGeometry>(target.geometry(path));
            current.observed_result.work.scheduled_positions = target.valid_target_tokens();
            current.cost_query = speculative::CostQuery::from_split_plan(
                speculative_backend, initial, target, frontiers, capacity);
        }
    }
    const auto top1_rows = current.round && draft_window && generated.row_stride == 1 ? batch : 0;
    const auto valid_probability = [](float p) { return p >= 0.0f && p <= 1.0f; };
    if (generated.draft_stride != stride || generated.draft_confidences.size() != batch * stride ||
        generated.target_top1_probabilities.size() != top1_rows ||
        !std::ranges::all_of(generated.draft_confidences, valid_probability) ||
        !std::ranges::all_of(generated.target_top1_probabilities, valid_probability))
        throw std::logic_error("generated probabilities differ from the decode round");
    PendingBatch out;
    out.ticket_    = current.ticket;
    out.members_   = current.members;
    out.tokens_    = generated.tokens;
    out.stride_    = generated.row_stride;
    out.execution_ = current.observed_result;
    for (std::size_t b = 0; b < batch; ++b) {
        const auto count   = generated.counts[b];
        const bool sampled = current.kind == OperationKind::Decode || current.sampled[b];
        if (sampled != (count != 0) || count > generated.row_stride || count > 16 ||
            (current.kind == OperationKind::Prefill && count > 1) ||
            (current.round && count - 1 > generated.executed->verify_drafts[b]))
            throw std::logic_error("generated result has an invalid licensed prefix");
        if (current.kind == OperationKind::Prefill) out.sampled_.push_back(sampled);
        out.counts_.push_back(static_cast<std::int32_t>(count));
        out.failed_.push_back((generated.dead_positions[b] & ((1U << count) - 1)) != 0);
        if (sampled) requests[lane(current.members[b])].lifecycle = Lifecycle::Pending;
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
            progress.rows.reserve(current.members.size());
            bool sampled = false;
            for (std::size_t b = 0; b < current.members.size(); ++b) {
                const auto& prefilling = requests[lane(current.members[b])];
                const auto frontier    = current.completed[b].execution_frontier;
                sampled |= current.sampled[b];
                progress.rows.push_back({
                    .summary = {prefilling.base->summary.prompt_tokens, prefilling.prefill->base,
                                prefilling.prefill->reuse},
                    .processed_prompt_tokens = current.columns[b],
                    .complete                = bool(current.sampled[b]),
                    .capture_ready           = prefilling.capture_reservation &&
                                     prefilling.capture_reservation->frontier == frontier,
                });
            }
            if (sampled != forward->generated.has_value())
                throw std::logic_error("prefill result changed its sampling rows");
            if (sampled) {
                current.generated.emplace(std::move(*forward->generated));
                progress.pending.emplace(generated_pending(*current.generated));
                events.push_back({current.ticket, current.members, std::move(progress)});
                return;
            }
            current.completion.emplace(
                SessionEvent{current.ticket, current.members, std::move(progress)});
        } else if (current.kind == OperationKind::Replay) {
            current.completion.emplace(
                SessionEvent{current.ticket, current.members,
                             ReplayProgress{capture, current.columns[0],
                                            end == request.replay_target && !capture, end}});
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
    std::vector<std::uint8_t> recovery(current.members.size());
    std::vector<context::PrefixUpdate> prefixes(current.members.size());
    Commit physical;
    physical.rows.reserve(current.members.size());
    for (std::size_t b = 0; b < current.members.size(); ++b) {
        const auto l        = lane(current.members[b]);
        auto& state         = sequences[l];
        const auto appended = current.kind == OperationKind::Control ? current.inputs[b].size() : 0;
        state.ledger.prepare_append(appended);
        if (!causal_scoring)
            prefixes[b] =
                state.prefix.prepare(requests[l].base->identity_view(state.ledger.view(
                                         appended ? std::span<const TokenId>(current.inputs[b])
                                                  : std::span<const TokenId>{})),
                                     current.completed[b].execution_frontier);
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
        if (current.kind == OperationKind::Control) state.ledger.append(current.inputs[b]);
        if (!causal_scoring) state.prefix.commit(std::move(prefixes[b]));
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
        if (!pending.row_sampled(b)) {
            if (d.accepted_tokens || d.failed || d.terminal != d.cancelled)
                throw std::logic_error("invalid unsampled prefill decision");
            continue;
        }
        if ((d.cancelled && d.failed) || (d.failed && !pending.failed_[b]) ||
            ((d.cancelled || d.failed)
                 ? (!d.terminal || d.accepted_tokens != 0)
                 : (!d.accepted_tokens ||
                    d.accepted_tokens > static_cast<std::uint32_t>(pending.counts_[b]) ||
                    (!d.terminal &&
                     d.accepted_tokens != static_cast<std::uint32_t>(pending.counts_[b])) ||
                    pending.failed_[b])))
            throw std::logic_error("invalid accepted-prefix decision");
    }
    CommitResult result;
    result.rows.resize(decisions.size());
    std::vector<std::uint8_t> recovery(current.members.size());
    std::vector<context::PrefixUpdate> prefixes(current.members.size());
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
            const auto physical_count =
                pending.row_sampled(b) || d.cancelled ? d.accepted_tokens : current.columns[b];
            physical.rows.push_back({physical_count, d.terminal, d.cancelled || d.failed});
            if (d.cancelled || d.failed) continue;
            state.ledger.prepare_append(d.accepted_tokens);
            const auto accepted = std::span<const TokenId>(pending.tokens_)
                                      .subspan(b * pending.stride_, d.accepted_tokens);
            prefixes[b] =
                state.prefix.prepare(requests[l].base->identity_view(state.ledger.view(accepted)),
                                     current.completed[b].execution_frontier);
            recovery[b] =
                prepare_settlement(l, current.completed[b],
                                   !pending.row_sampled(b) ? Lifecycle::Prefilling
                                   : d.terminal            ? Lifecycle::Finishable
                                                           : Lifecycle::Active,
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
            state.ledger.append(accepted);
            state.prefix.commit(std::move(prefixes[b]));
            request.lifecycle = !pending.row_sampled(b) ? Lifecycle::Prefilling
                                : d.terminal            ? Lifecycle::Finishable
                                                        : Lifecycle::Active;
            request.capture_pending =
                request.capture_reservation &&
                request.capture_reservation->frontier == current.completed[b].execution_frontier;
        }
        if (d.terminal) {
            request.capture_pending = false;
            request.capture_reservation.reset();
            request.prefill.reset();
        } else if (current.kind == OperationKind::Prefill && pending.row_sampled(b) &&
                   !request.capture_pending) {
            request.prefill.reset();
        }
        auto& stats = request.speculative_stats;
        if (current.kind == OperationKind::Decode && draft_window &&
            speculative_backend != SpeculativeBackend::None) {
            const auto& generated = *current.generated;
            const auto drafted    = generated.executed->verify_drafts[b];
            const auto proposed   = current.round->verify_drafts()[b];
            const auto confidences =
                drafted ? std::span<const float>(generated.draft_confidences)
                              .subspan(b * generated.draft_stride, generated.draft_stride)
                        : std::span<const float>{};
            auto* feedback = plan.length_policy == SpeculativeLengthPolicy::Adaptive
                                 ? &request.speculative_state
                                 : nullptr;
            speculative::record_speculative_commit(
                stats, feedback, d, current.could_speculate[b], proposed, drafted,
                generated.counts[b] - 1, confidences,
                generated.target_top1_probabilities.empty()
                    ? std::nullopt
                    : std::optional{generated.target_top1_probabilities[b]});
        }
        auto& row         = result.rows[b];
        row.disposition   = d.cancelled  ? runtime::CommitDisposition::CancelledReleased
                            : d.failed   ? runtime::CommitDisposition::FailedReleased
                            : d.terminal ? runtime::CommitDisposition::Finishable
                                         : runtime::CommitDisposition::Active;
        row.speculative   = stats;
        row.capture_ready = !d.terminal && request.capture_pending;
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
    for (auto h : current.members) sequences[lane(h)].computing = false;
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
    if (current.kind == OperationKind::Decode && current.generated)
        event.execution.work = current.observed_result.work;
    if (current.kind != OperationKind::Normalize && current.kind != OperationKind::Score)
        events.push_back({current.ticket, current.members,
                          runtime::ExecutionComplete{event.execution, current.members.size()}});
    if (current.cost_query && current.record_online_cost && !current.discarded &&
        std::ranges::none_of(current.decisions,
                             [](const auto& d) { return d.cancelled || d.failed; }))
        cost_model.observe_retired_step(*current.cost_query,
                                        double(event.execution.latency.total_ns) * 1.0e-6);
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
