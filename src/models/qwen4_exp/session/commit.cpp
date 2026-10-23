#include "models/qwen4_exp/session/internal.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp::detail {
namespace {
using Kind = ComputeTransaction::Kind;

bool reached_dead_end(const RowResult& row) {
    const auto count   = row.tokens.size();
    const auto reached = count >= 32 ? UINT32_MAX : (std::uint32_t{1} << count) - 1;
    return (row.dead_positions & reached) != 0;
}

void append_identity(Sequence& s, std::span<const TokenId> tokens,
                     std::optional<std::uint32_t> split) noexcept {
    s.identity.tokens.insert(s.identity.tokens.end(), tokens.begin(), tokens.end());
    s.identity.positions.append_generated(tokens.size(), s.continuation.metadata.rope_delta, split);
    s.identity.digests.append_generated(tokens, s.continuation.metadata.rope_delta, split);
}
} // namespace

void SessionImpl::finish_result(ComputeResult result) {
    auto& tx = *compute;
    if (result.rows.size() != tx.members.size())
        throw std::logic_error("Qwen4Exp result changed its membership");
    tx.result.emplace(std::move(result));
    bool generated = false;
    for (std::size_t b = 0; b < tx.members.size(); ++b) {
        const auto& row = tx.result->rows[b];
        generated |= !row.tokens.empty();
        if (tx.kind != Kind::Mtp && row.processed != tx.rows[b].ids.size())
            throw std::logic_error("Qwen4Exp result changed its input extent");
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
        for (std::size_t b = 0; b < tx.members.size(); ++b) {
            const auto& row = tx.result->rows[b];
            if (row.tokens.empty()) throw std::logic_error("sampled batch has an empty row");
            batch.counts_[b] = row.tokens.size();
            std::copy(row.tokens.begin(), row.tokens.end(),
                      batch.tokens_.begin() + b * batch.stride_);
            batch.failed_[b]              = reached_dead_end(row);
            sequence(tx.members[b]).phase = Phase::Pending;
        }
        if (tx.kind == Kind::Prefill) {
            const auto& s = sequence(tx.members[0]);
            PrefillProgress progress;
            progress.summary                 = {s.request->summary.prompt_tokens, s.reused,
                                s.reused ? PrefixReusePath::Checkpoint : PrefixReusePath::Root};
            progress.processed_prompt_tokens = tx.result->rows[0].processed;
            progress.complete                = true;

            progress.pending.emplace(std::move(batch));
            events.push_back({tx.ticket, tx.members, std::move(progress)});
        } else {
            events.push_back({tx.ticket, tx.members, std::move(batch)});
        }
        return;
    }

    tx.completion = std::make_unique<context::ContextTransaction>(directory);
    Commit commit;
    std::array<ContinuationMetadata, kMaximumConcurrency> completed{};
    commit.rows.reserve(tx.members.size());
    const bool retain_recovery = tx.kind == Kind::Replay;
    for (std::size_t b = 0; b < tx.members.size(); ++b) {
        auto& s         = sequence(tx.members[b]);
        const auto& row = tx.result->rows[b];
        commit.rows.push_back({member(s), tx.rows[b].states, row.processed});
        if (tx.kind == Kind::Control) {
            const auto count = tx.rows[b].count_tokens.size();
            s.identity.tokens.reserve(s.identity.tokens.size() + count);
            s.identity.positions.prepare_generated(count, s.continuation.metadata.rope_delta,
                                                   tx.splits[b]);
            s.identity.digests.prepare_generated(count, s.continuation.metadata.rope_delta,
                                                 tx.splits[b]);
        }
        completed[b] =
            prepare_commit(s, tx.rows[b], tx.rows[b].ids, retain_recovery, *tx.completion);
    }
    if (tx.kind == Kind::Prefill) {
        const auto& s = sequence(tx.members[0]);
        PrefillProgress progress;
        progress.summary                 = {s.request->summary.prompt_tokens, s.reused,
                            s.reused ? PrefixReusePath::Checkpoint : PrefixReusePath::Root};
        progress.processed_prompt_tokens = tx.result->rows[0].processed;
        progress.complete                = false;
        progress.capture_ready =
            s.capture && s.captures[s.next_capture].frontier == completed[0].main_frontier;

        tx.event.emplace(SessionEvent{tx.ticket, tx.members, std::move(progress)});
    } else if (tx.kind == Kind::Replay) {
        const auto& s = sequence(tx.members[0]);
        tx.event.emplace(
            SessionEvent{tx.ticket, tx.members,
                         ReplayProgress{false, tx.result->rows[0].processed,
                                        completed[0].main_frontier == s.replay_target}});
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
        if (tx.kind == Kind::Control) append_identity(s, tx.rows[b].count_tokens, tx.splits[b]);
        if (tx.kind == Kind::Replay && s.continuation.metadata.main_frontier == s.replay_target)
            s.phase = s.resume_phase;
    }
}

void SessionImpl::retire_compute(const runtime::DeviceRetired& event) {
    auto& tx = *compute;
    tx.context.mark_completed();
    tx.context.publish();
    if (tx.completion) {
        tx.completion->mark_completed();
        tx.completion->publish();
    }
    if (tx.kind != Kind::Score)
        events.push_back({tx.ticket, tx.members,
                          runtime::ExecutionComplete{event.execution, tx.members.size()}});
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

} // namespace ninfer::models::qwen4_exp::detail

namespace ninfer::models::qwen4_exp {

CommitResult Session::commit(PendingBatch&& batch,
                             std::span<const runtime::CommitDecision> decisions) {
    auto& p = *impl_;
    if (!p.compute || batch.ticket_ != p.compute->ticket || batch.members_ != p.compute->members ||
        !p.compute->result || p.compute->committed || decisions.size() != batch.row_count())
        throw std::logic_error("invalid pending Qwen4Exp transaction");
    auto& tx = *p.compute;
    for (std::size_t b = 0; b < decisions.size(); ++b) {
        const auto& decision = decisions[b];
        const auto produced  = tx.result->rows[b].tokens.size();
        if ((decision.cancelled && decision.failed) || (decision.failed && !batch.failed_[b]) ||
            ((decision.cancelled || decision.failed)
                 ? (!decision.terminal || decision.accepted_tokens != 0)
                 : (!decision.accepted_tokens || decision.accepted_tokens > produced ||
                    (!decision.terminal && decision.accepted_tokens != produced) ||
                    batch.failed_[b])) ||
            (decision.prefix_execution_split_after &&
             (!*decision.prefix_execution_split_after ||
              *decision.prefix_execution_split_after > decision.accepted_tokens)))
            throw std::logic_error("invalid generated-prefix decision");
    }
    CommitResult result;
    result.row_count = decisions.size();
    std::array<ContinuationMetadata, kMaximumConcurrency> completed{};
    try {
        tx.decisions.assign(decisions.begin(), decisions.end());
        tx.completion = std::make_unique<context::ContextTransaction>(p.directory);
        Commit physical;
        physical.rows.reserve(decisions.size());
        for (std::size_t b = 0; b < decisions.size(); ++b) {
            auto& s              = p.sequence(tx.members[b]);
            const auto& decision = decisions[b];
            result.rows[b].speculative.accepted_per_position.reserve(
                s.speculative.accepted_per_position.size());
            physical.rows.push_back({p.member(s), tx.rows[b].states, decision.accepted_tokens,
                                     decision.terminal, decision.cancelled, decision.failed});
            if (!decision.accepted_tokens) continue;
            const auto count = decision.accepted_tokens;
            s.identity.tokens.reserve(s.identity.tokens.size() + count);
            s.identity.positions.prepare_generated(count, s.continuation.metadata.rope_delta,
                                                   decision.prefix_execution_split_after);
            s.identity.digests.prepare_generated(count, s.continuation.metadata.rope_delta,
                                                 decision.prefix_execution_split_after);
            if (tx.kind == detail::ComputeTransaction::Kind::Mtp) {
                const auto& inputs = tx.result->rows[b].input_tokens;
                if (inputs.size() < count)
                    throw std::logic_error("MTP result omitted its adopted input prefix");
                completed[b] =
                    p.prepare_commit(s, tx.rows[b], std::span<const TokenId>(inputs).first(count),
                                     false, *tx.completion);
            } else {
                completed[b] =
                    p.prepare_commit(s, tx.rows[b], tx.rows[b].ids, false, *tx.completion);
            }
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
        if (batch.decode_ && p.definition->options().mtp()) {
            auto& stats         = s.speculative;
            const auto drafted  = tx.result->rows[b].drafts;
            const auto accepted = tx.result->rows[b].accepted_drafts;
            stats.committed_draft_tokens += std::min(decision.accepted_tokens, accepted);
            if (drafted)
                ++stats.rounds;
            else
                ++stats.fallback_steps;
            stats.drafted_tokens += drafted;
            stats.accepted_tokens += accepted;
            for (std::uint32_t i = 0; i < accepted; ++i) ++stats.accepted_per_position[i];
        }
        const auto& stats = s.speculative;
        row.speculative   = stats;

        if (decision.cancelled || decision.failed) {
            row.disposition = decision.cancelled ? runtime::CommitDisposition::CancelledReleased
                                                 : runtime::CommitDisposition::FailedReleased;
        } else {
            p.adopt_commit(s, completed[b],
                           tx.kind == detail::ComputeTransaction::Kind::Mtp ||
                               tx.result->rows[b].processed != 0,
                           false);
            if (tx.kind == detail::ComputeTransaction::Kind::Prefill)
                s.capture_pending =
                    s.capture && s.captures[s.next_capture].frontier == completed[b].main_frontier;
            const auto tokens =
                std::span<const TokenId>(tx.result->rows[b].tokens).first(decision.accepted_tokens);
            detail::append_identity(s, tokens, decision.prefix_execution_split_after);
            s.phase         = decision.terminal ? detail::Phase::Finished : detail::Phase::Active;
            row.disposition = decision.terminal ? runtime::CommitDisposition::Finishable
                                                : runtime::CommitDisposition::Active;
            result.capture_ready[b] = s.capture_pending;
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

} // namespace ninfer::models::qwen4_exp
