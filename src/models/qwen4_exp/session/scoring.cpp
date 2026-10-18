#include "models/qwen4_exp/session/internal.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

std::optional<runtime::ExecutionTicket> Session::start_score(qwen::PreparedPrompt&& prompt,
                                                             std::uint32_t first_target) {
    auto& p = *impl_;
    if (p.shape.purpose != EnginePurpose::CausalScoring)
        throw std::logic_error("not a causal scoring Session");
    if (p.compute || p.operation || p.scoring || p.release_ticket || !p.releases.empty()) return {};
    runtime::ResolvedExecutionOptions options;
    options.allow_prefix_reuse = false;
    auto base                  = plan_request(std::move(prompt), options);
    if (!first_target || first_target >= base.summary().prompt_tokens)
        throw std::invalid_argument("invalid scoring target range");
    auto scoring          = std::make_unique<detail::Scoring>();
    scoring->first_target = first_target;
    scoring->logprobs.reserve(base.summary().prompt_tokens - first_target);
    p.scoring = std::move(scoring);
    try {
        if (!start_binding(base, {0}, {})) {
            p.scoring.reset();
            return {};
        }
        p.scoring->ticket = *p.operation->ticket;
        return p.scoring->ticket;
    } catch (...) {
        p.scoring.reset();
        throw;
    }
}

} // namespace ninfer::models::qwen4_exp

namespace ninfer::models::qwen4_exp::detail {

bool SessionImpl::submit_score() {
    if (!scoring || !scoring->sequence || compute)
        throw std::logic_error("scoring continuation is unavailable");
    auto& s          = sequence(*scoring->sequence);
    const auto begin = s.continuation.metadata.main_frontier;
    const auto total = s.request->summary.prompt_tokens;
    if (begin + 1 >= total) {
        if (scoring->logprobs.size() != total - scoring->first_target)
            throw std::logic_error("scoring result omitted targets");
        const auto lane = s.lane;
        clear(s);
        sequences[lane].reset();
        scoring->sequence.reset();
        scoring->finishing = true;
        submit_releases();
        return true;
    }
    const auto end   = window_end(s, begin, total - 1);
    auto reservation = executor->try_reserve(runtime::FrameKind::Compute);
    if (!reservation) return false;
    compute  = std::make_unique<ComputeTransaction>(ComputeTransaction::Kind::Score, directory);
    auto& tx = *compute;
    tx.members.push_back(handle(s));
    try {
        for (std::uint32_t i = 0; i < pool_count(); ++i) {
            auto& histories  = directory.pool(i).histories();
            const auto first = i ? s.continuation.metadata.mtp_frontier.value() : begin;
            const auto last  = i ? mtp_history_frontier(end) : end;
            histories.reserve_growth(s.histories[i].handle(),
                                     histories.write_pages(s.histories[i].handle(), first, last));
        }
        prepare_write(s, end, tx.context);
        ScoreWork work;
        work.row                 = work_row(s, end - begin);
        const auto first         = std::min(end, std::max(begin, scoring->first_target - 1));
        work.first_target_column = first - begin;
        work.targets.assign(s.identity.tokens.begin() + first + 1,
                            s.identity.tokens.begin() + end + 1);
        tx.rows.push_back(work.row);
        tx.context.prepare_actions();
        tx.ticket =
            executor->submit(std::move(*reservation), {tx.context.actions(), std::move(work)});
        tx.context.mark_submitted();
        return true;
    } catch (const std::exception& error) {
        executor->fail(error.what());
        throw;
    }
}

void SessionImpl::finish_score(ScoreResult result) {
    auto& tx = *compute;
    if (!scoring || tx.kind != ComputeTransaction::Kind::Score || tx.rows.size() != 1 ||
        result.processed != tx.rows[0].ids.size())
        throw std::logic_error("invalid Qwen4Exp score result");
    auto& s          = sequence(tx.members[0]);
    const auto begin = tx.rows[0].main_frontier;
    const auto end   = begin + result.processed;
    const auto first = std::min(end, std::max(begin, scoring->first_target - 1));
    if (result.logprobs.size() != end - first)
        throw std::logic_error("score result has the wrong target extent");
    if (result.logprobs.size() > scoring->logprobs.capacity() - scoring->logprobs.size())
        throw std::logic_error("score result exceeds its prepared target capacity");
    tx.completion        = std::make_unique<context::ContextTransaction>(directory);
    const auto completed = prepare_commit(s, tx.rows[0], tx.rows[0].ids, false, *tx.completion);
    tx.completion->prepare_actions();
    Commit commit;
    commit.rows.push_back({member(s), tx.rows[0].slots, result.processed});
    commit.actions = tx.completion->actions();
    (void)executor->commit(tx.ticket, std::move(commit));
    tx.completion->mark_submitted();
    tx.completion->adopt();
    adopt_commit(s, completed, result.processed != 0, false);
    scoring->logprobs.insert(scoring->logprobs.end(), result.logprobs.begin(),
                             result.logprobs.end());
    tx.committed = true;
}

} // namespace ninfer::models::qwen4_exp::detail
