#include "models/glm5next/session/internal.h"

#include <algorithm>

namespace ninfer::models::glm5next::detail {
namespace {
using Kind = ComputeTransaction::Kind;

bool reached_dead_end(const RowResult& row) {
    const auto count   = row.tokens.size();
    const auto reached = count >= 32 ? UINT32_MAX : (std::uint32_t{1} << count) - 1;
    return (row.dead_positions & reached) != 0;
}
} // namespace

void SessionImpl::finish_result(ComputeResult result) {
    auto& tx = *compute;
    if (result.rows.size() != tx.members.size())
        throw std::logic_error("GLM5Next result changed its membership");
    tx.result.emplace(std::move(result));
    const bool prefill = tx.kind == Kind::Prefill;
    PrefillProgress progress;
    if (prefill) progress.rows.reserve(tx.members.size());
    bool generated = false;
    for (std::size_t b = 0; b < tx.members.size(); ++b) {
        const auto& row    = tx.result->rows[b];
        const bool sampled = !row.tokens.empty();
        generated |= sampled;
        if (prefill) {
            const auto& work = tx.rows[b];
            const auto& s    = sequence(tx.members[b]);
            const auto end   = work.main_frontier + row.processed;
            if (sampled != work.sample || (sampled && row.tokens.size() != 1))
                throw std::logic_error("prefill result changed its sampling rows");
            progress.rows.push_back({
                .summary                 = {s.request->summary.prompt_tokens, s.reused,
                            s.reused ? PrefixReusePath::Checkpoint : PrefixReusePath::Root},
                .processed_prompt_tokens = row.processed,
                .complete                = sampled,
                .capture_ready           = s.capture && s.captures[s.next_capture].frontier == end,
            });
        }
        if (tx.kind == Kind::Speculative
                ? (row.drafts != tx.round->verify_drafts()[b] ||
                   row.processed != tx.round->verify_drafts()[b] + 1 ||
                   row.accepted_drafts > row.drafts || row.tokens.size() != row.accepted_drafts + 1)
                : (row.processed != tx.rows[b].ids.size()))
            throw std::logic_error("GLM5Next result changed its input extent");
    }
    if (generated) {
        PendingBatch batch;
        batch.ticket_    = tx.ticket;
        batch.members_   = tx.members;
        batch.decode_    = tx.kind != Kind::Prefill;
        batch.stride_    = 1;
        batch.execution_ = tx.observed_result;
        for (const auto& row : tx.result->rows)
            batch.stride_ = std::max<std::uint32_t>(batch.stride_, row.tokens.size());
        batch.tokens_.resize(batch.stride_ * tx.members.size());
        batch.counts_.resize(tx.members.size());
        batch.failed_.resize(tx.members.size());
        if (prefill) batch.sampled_.resize(tx.members.size());
        for (std::size_t b = 0; b < tx.members.size(); ++b) {
            const auto& row    = tx.result->rows[b];
            const bool sampled = !row.tokens.empty();
            if (!prefill && !sampled) throw std::logic_error("decode has an empty sampled row");
            if (prefill) batch.sampled_[b] = sampled;
            batch.counts_[b] = row.tokens.size();
            std::copy(row.tokens.begin(), row.tokens.end(),
                      batch.tokens_.begin() + b * batch.stride_);
            batch.failed_[b]              = reached_dead_end(row);
            sequence(tx.members[b]).phase = Phase::Pending;
        }
        if (prefill) {
            progress.pending.emplace(std::move(batch));
            events.push_back({tx.ticket, tx.members, std::move(progress)});
        } else {
            events.push_back({tx.ticket, tx.members, std::move(batch)});
        }
        return;
    }

    tx.completion = std::make_unique<context::ContextTransaction>(directory);
    Commit commit;
    std::vector<ContinuationMetadata> completed(tx.members.size());
    std::vector<context::PrefixUpdate> prefixes(tx.members.size());
    commit.rows.reserve(tx.members.size());
    const bool retain_recovery = tx.kind == Kind::Replay;
    for (std::size_t b = 0; b < tx.members.size(); ++b) {
        auto& s         = sequence(tx.members[b]);
        const auto& row = tx.result->rows[b];
        commit.rows.push_back({member(s), tx.rows[b].states, row.processed});
        if (tx.kind == Kind::Control) {
            const auto count = tx.rows[b].ids.size();
            s.ledger.prepare_append(count);
        }
        completed[b] =
            prepare_commit(s, tx.rows[b], tx.rows[b].ids.size(), retain_recovery, *tx.completion);
        const auto staged = tx.kind == Kind::Control
                                ? std::span<const TokenId>(tx.rows[b].ids).subspan(1)
                                : std::span<const TokenId>{};
        prefixes[b]       = s.prefix.prepare(s.request->identity_view(s.ledger.view(staged)),
                                             completed[b].main_frontier);
    }
    if (prefill) {
        tx.event.emplace(SessionEvent{tx.ticket, tx.members, std::move(progress)});
    } else if (tx.kind == Kind::Replay) {
        const auto& s = sequence(tx.members[0]);
        tx.event.emplace(SessionEvent{tx.ticket, tx.members,
                                      ReplayProgress{false, tx.result->rows[0].processed,
                                                     completed[0].main_frontier == s.replay_target,
                                                     completed[0].main_frontier}});
    } else if (tx.kind == Kind::Control) {
        tx.event.emplace(SessionEvent{tx.ticket, tx.members, ControlComplete{}});
    } else {
        throw std::logic_error("decode completed without generated tokens");
    }
    tx.completion->prepare_actions();
    commit.actions = actions(*tx.completion);
    (void)executor->commit(tx.ticket, std::move(commit));
    tx.completion->mark_submitted();
    tx.completion->adopt();
    tx.committed = true;
    for (std::size_t b = 0; b < tx.members.size(); ++b) {
        auto& s = sequence(tx.members[b]);
        adopt_commit(s, completed[b], tx.result->rows[b].processed != 0, retain_recovery);
        if (tx.kind == Kind::Prefill)
            s.capture_pending =
                s.capture && s.captures[s.next_capture].frontier == completed[b].main_frontier;
        if (tx.kind == Kind::Control) {
            const auto& row   = tx.rows[b];
            const auto prefix = std::span<const TokenId>(row.ids).subspan(1);
            s.ledger.append(prefix);
            s.ledger.append({&*row.control_last_token, 1});
        }
        s.prefix.commit(std::move(prefixes[b]));
        if (tx.kind == Kind::Replay && s.continuation.metadata.main_frontier == s.replay_target)
            s.phase = s.resume_phase;
    }
}

void SessionImpl::retire_compute(const runtime::DeviceRetired& event) {
    auto& tx = *compute;
    for (auto h : tx.members) sequence(h).computing = false;
    if (tx.vision_id) {
        const auto* window = vision_window.current();
        bool complete      = tx.discarded;
        if (window && valid(window->owner.sequence))
            complete |= sequence(window->owner.sequence).continuation.metadata.main_frontier >=
                        window->plan->consume_end;
        else
            complete = true;
        vision_window.consumed(tx.vision_id, complete);
        vision_window.retired(event.ticket);
    }
    tx.context.mark_completed();
    tx.context.publish();
    if (tx.completion) {
        tx.completion->mark_completed();
        tx.completion->publish();
    }
    if (tx.kind != Kind::Score)
        events.push_back({tx.ticket, tx.members,
                          runtime::ExecutionComplete{event.execution, tx.members.size()}});
    if (tx.cost_query && tx.record_online_cost && !tx.discarded &&
        std::ranges::none_of(tx.decisions, [](const auto& d) { return d.cancelled || d.failed; }))
        cost_model.observe_retired_step(*tx.cost_query,
                                        double(event.execution.latency.total_ns) * 1.0e-6);
    for (std::size_t b = 0; b < tx.members.size(); ++b) {
        if (!tx.discarded &&
            (tx.decisions.empty() || (!tx.decisions[b].cancelled && !tx.decisions[b].failed)))
            continue;
        auto& s         = sequence(tx.members[b]);
        const auto lane = s.lane;
        clear(s);
        sequences[lane].reset();
    }
    if (tx.event) events.push_back(std::move(*tx.event));
    const bool score = tx.kind == Kind::Score;
    compute.reset();
    if (score && scoring) (void)submit_score();
}

} // namespace ninfer::models::glm5next::detail

namespace ninfer::models::glm5next {

CommitResult Session::commit(PendingBatch&& batch,
                             std::span<const runtime::CommitDecision> decisions) {
    auto& p = *impl_;
    if (!p.compute || batch.ticket_ != p.compute->ticket || batch.members_ != p.compute->members ||
        !p.compute->result || p.compute->committed || decisions.size() != batch.row_count())
        throw std::logic_error("invalid pending GLM5Next transaction");
    auto& tx = *p.compute;
    for (std::size_t b = 0; b < decisions.size(); ++b) {
        const auto& decision = decisions[b];
        const auto produced  = tx.result->rows[b].tokens.size();
        if (!batch.row_sampled(b)) {
            if (decision.accepted_tokens || decision.failed ||
                decision.terminal != decision.cancelled)
                throw std::logic_error("invalid unsampled prefill decision");
            continue;
        }
        if ((decision.cancelled && decision.failed) || (decision.failed && !batch.failed_[b]) ||
            ((decision.cancelled || decision.failed)
                 ? (!decision.terminal || decision.accepted_tokens != 0)
                 : (!decision.accepted_tokens || decision.accepted_tokens > produced ||
                    (!decision.terminal && decision.accepted_tokens != produced) ||
                    batch.failed_[b])))
            throw std::logic_error("invalid generated-prefix decision");
    }
    CommitResult result;
    result.rows.resize(decisions.size());
    std::vector<ContinuationMetadata> completed(decisions.size());
    std::vector<context::PrefixUpdate> prefixes(decisions.size());
    try {
        tx.decisions.assign(decisions.begin(), decisions.end());
        tx.completion = std::make_unique<context::ContextTransaction>(p.directory);
        Commit physical;
        physical.rows.reserve(decisions.size());
        for (std::size_t b = 0; b < decisions.size(); ++b) {
            auto& s              = p.sequence(tx.members[b]);
            const auto& decision = decisions[b];
            result.rows[b].speculative.attempted_per_position.reserve(
                s.speculative.attempted_per_position.size());
            result.rows[b].speculative.accepted_per_position.reserve(
                s.speculative.accepted_per_position.size());
            const auto physical_count = batch.row_sampled(b) || decision.cancelled
                                            ? decision.accepted_tokens
                                            : static_cast<std::uint32_t>(tx.rows[b].ids.size());
            physical.rows.push_back({p.member(s), tx.rows[b].states, physical_count,
                                     decision.terminal, decision.cancelled, decision.failed});
            if (decision.cancelled || decision.failed) continue;
            const auto count = decision.accepted_tokens;
            s.ledger.prepare_append(count);
            const auto inputs = tx.kind == detail::ComputeTransaction::Kind::Speculative
                                    ? count
                                    : tx.rows[b].ids.size();
            completed[b]      = p.prepare_commit(s, tx.rows[b], inputs, false, *tx.completion);
        }
        for (std::size_t b = 0; b < decisions.size(); ++b) {
            if (decisions[b].cancelled || decisions[b].failed) continue;
            auto& s             = p.sequence(tx.members[b]);
            const auto accepted = std::span<const TokenId>(tx.result->rows[b].tokens)
                                      .first(decisions[b].accepted_tokens);
            prefixes[b] = s.prefix.prepare(s.request->identity_view(s.ledger.view(accepted)),
                                           completed[b].main_frontier);
        }
        tx.completion->prepare_actions();
        physical.actions = p.actions(*tx.completion);
        (void)p.executor->commit(tx.ticket, std::move(physical));
        tx.completion->mark_submitted();
        tx.completion->adopt();
        tx.committed = true;
    } catch (const std::exception& error) {
        p.executor->fail(error.what());
        throw;
    }
    for (std::size_t b = 0; b < decisions.size(); ++b) {
        auto& s              = p.sequence(tx.members[b]);
        const auto& decision = decisions[b];
        auto& row            = result.rows[b];
        if (batch.decode_ && p.shape.draft_tokens) {
            const auto& output = tx.result->rows[b];
            auto* feedback     = p.shape.length_policy == SpeculativeLengthPolicy::Adaptive
                                     ? &s.speculative_state
                                     : nullptr;
            speculative::record_speculative_commit(
                s.speculative, feedback, decision, tx.could_speculate[b], output.drafts,
                output.drafts, output.accepted_drafts, output.draft_confidences,
                output.target_top1_probability);
        }
        row.speculative = s.speculative;

        if (decision.cancelled || decision.failed) {
            row.disposition = decision.cancelled ? runtime::CommitDisposition::CancelledReleased
                                                 : runtime::CommitDisposition::FailedReleased;
        } else {
            p.adopt_commit(s, completed[b], tx.result->rows[b].processed != 0, false);
            if (tx.kind == detail::ComputeTransaction::Kind::Prefill)
                s.capture_pending =
                    s.capture && s.captures[s.next_capture].frontier == completed[b].main_frontier;
            const auto tokens =
                std::span<const TokenId>(tx.result->rows[b].tokens).first(decision.accepted_tokens);
            s.ledger.append(tokens);
            s.prefix.commit(std::move(prefixes[b]));
            s.phase         = !batch.row_sampled(b) ? detail::Phase::Prefill
                              : decision.terminal   ? detail::Phase::Finished
                                                    : detail::Phase::Active;
            row.disposition = decision.terminal ? runtime::CommitDisposition::Finishable
                                                : runtime::CommitDisposition::Active;
            result.rows[b].capture_ready = s.capture_pending;
        }
    }
    batch.ticket_ = {};
    return result;
}

DiscardResult Session::discard(PendingBatch&& batch) noexcept {
    auto& p = *impl_;
    if (!p.compute || batch.ticket_ != p.compute->ticket || batch.members_ != p.compute->members ||
        p.compute->committed)
        return {};
    try {
        p.executor->discard(p.compute->ticket);
        p.compute->discarded = true;
        p.compute->committed = true;
        batch.ticket_        = {};
        return {runtime::ConsumeStatus::Consumed, batch.row_count()};
    } catch (...) { return {}; }
}

FinishResult Session::finish(SequenceHandle handle) {
    auto& p = *impl_;
    if (!p.valid(handle) || execution_pending(handle) || context_blocks(handle)) return {};
    auto& s = p.sequence(handle);
    if (s.phase != detail::Phase::Finished) return {};
    FinishResult result{runtime::ConsumeStatus::Consumed, s.speculative, {}};
    if (s.request->summary.publish_continuation)
        result.checkpoint = p.detach(s, runtime::CheckpointRole::Continuation);
    const auto lane = s.lane;
    p.clear(s);
    p.sequences[lane].reset();
    return result;
}

AbortResult Session::abort(SequenceHandle handle) {
    auto& p = *impl_;
    if (!p.valid(handle) || execution_pending(handle) || context_blocks(handle)) return {};
    auto& s = p.sequence(handle);
    AbortResult result{runtime::ConsumeStatus::Consumed, s.speculative};
    const auto lane = s.lane;
    p.clear(s);
    p.sequences[lane].reset();
    return result;
}

} // namespace ninfer::models::glm5next
