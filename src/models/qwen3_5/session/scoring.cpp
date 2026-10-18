#include "models/qwen3_5/session/internal.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {

std::optional<runtime::ExecutionTicket> SessionImpl::start_score(qwen::PreparedPromptData&& prompt,
                                                                 std::uint32_t first_target) {
    if (!causal_scoring) throw std::logic_error("Session is not configured for causal scoring");
    if (scoring || execution || has_context_operation()) return {};
    if (prompt.token_ids.size() < 2 || prompt.token_ids.size() > capacity || !first_target ||
        first_target >= prompt.token_ids.size())
        throw std::invalid_argument("causal scoring window or target is invalid");
    if (prompt.has_media()) throw std::invalid_argument("causal scoring accepts text tokens only");
    runtime::ResolvedExecutionOptions options;
    options.allow_prefix_reuse = false;
    auto base                  = plan_request(std::move(prompt), options);
    const auto source          = inspect_source(base, {});
    if (!source) throw std::logic_error("scoring has no root binding");
    scoring.emplace(Scoring{.base = std::move(base), .first_target = first_target});
    try {
        scoring->logprobs.reserve(scoring->base.summary().prompt_tokens - first_target);
        const auto binding = start_binding(scoring->base, {0}, *source);
        if (!binding) {
            scoring.reset();
            return {};
        }
        if (!context_operation_ || !context_operation_->ticket)
            throw std::logic_error("scoring context was not submitted");
        scoring->public_ticket = *context_operation_->ticket;
        return scoring->public_ticket;
    } catch (...) {
        if (context_operation_ && context_operation_->ticket)
            executor().fail("scoring setup failed after context submission");
        else {
            abort_context();
            scoring.reset();
        }
        throw;
    }
}

void SessionImpl::advance_score() {
    if (!scoring || execution) throw std::logic_error("scoring continuation has no free frame");
    const auto handle  = scoring->sequence;
    auto& state        = sequences[lane(handle)];
    auto& request      = requests[state.lane];
    const auto& prompt = *scoring->base.impl_->prompt;
    const auto limit   = static_cast<std::uint32_t>(prompt.token_ids.size()) - 1;
    if (scoring->cursor == limit) {
        auto done = std::move(*scoring);
        scoring.reset();
        clear_lane(state, request);
        after_releases.emplace(SessionEvent{
            done.public_ticket, {}, ScoreComplete{std::move(done.logprobs), done.timing}});
        submit_releases();
        return;
    }
    request.prefill->cursor = scoring->cursor;
    const std::array unit{ExecutionUnit{handle, ExecutionUnitKind::Prefill, 0}};
    if (!reserve_units(unit)) throw std::logic_error("scoring lost its resident context capacity");
    const auto end   = std::min(limit, scoring->cursor + prefill_chunk);
    auto reservation = executor().try_reserve(runtime::FrameKind::Compute);
    if (!reservation) throw std::logic_error("scoring frame did not retire");
    try {
        Execution pending{.kind = OperationKind::Score};
        pending.members     = {handle};
        pending.bases       = {state.continuation};
        pending.columns     = {end - scoring->cursor};
        pending.preparation = std::make_unique<context::ContextTransaction>(directory);
        ensure_sequence_kv_mapped(state, end, 0, *pending.preparation);
        ScoreWork score;
        score.row        = execution_row(state);
        score.input      = input_window(state, scoring->cursor, end);
        const auto first = std::max(scoring->first_target, scoring->cursor + 1);
        if (first <= end) {
            score.first_column = first - scoring->cursor - 1;
            score.targets.assign(prompt.token_ids.begin() + first,
                                 prompt.token_ids.begin() + end + 1);
        }
        (void)submit_execution(std::move(pending), std::move(*reservation),
                               Work{.operation = std::move(score)});
    } catch (const std::exception& error) {
        executor().fail(error.what());
        throw;
    }
}

} // namespace ninfer::models::qwen3_5::detail
