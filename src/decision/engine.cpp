#include "decision/decision.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace ninfer::decision {

Json DecisionEngine::run(const Request& request, const PreparationControl& control) {
    const auto check = [&] {
        if (control.cancellation.requested()) {
            throw RequestError(RequestErrorKind::Cancelled, "decision request cancelled");
        }
        if (control.deadline != std::chrono::steady_clock::time_point{} &&
            std::chrono::steady_clock::now() >= control.deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "decision request deadline exceeded");
        }
    };
    Backend backend;
    backend.check_cancelled = check;
    backend.tokenize        = [&](const std::string& system, const std::string& context,
                                  const std::string& continuation) {
        check();
        PromptInput input;
        input.options.enable_thinking = false;
        input.options.continuation    = PromptContinuationMode::ContinueFinalAssistant;
        input.messages                = {
            {.role = ChatRole::System, .parts = {{.text = system}}},
            {.role = ChatRole::User, .parts = {{.text = context}}},
            {.role = ChatRole::Assistant, .parts = {{.text = continuation}}},
        };
        auto preparation     = control;
        const auto deadline  = std::chrono::steady_clock::now() +
                               std::chrono::milliseconds(engine_.options().pending_timeout_ms);
        preparation.deadline = control.deadline == std::chrono::steady_clock::time_point{}
                                   ? deadline
                                   : std::min(deadline, control.deadline);
        return engine_.tokenize_prompt(std::move(input), preparation);
    };
    backend.score = [&](std::vector<ScoreRow> rows, bool cache) {
        check();
        cache = cache && engine_.options().context_cache.enabled;
        std::vector<BranchScoreRow> native;
        native.reserve(rows.size());
        for (const auto& row : rows) {
            native.push_back(
                {row.prefix, row.suffix, row.candidates,
                 cache && !row.cache_frontiers.empty() ? row.cache_frontiers.back() : 0,
                 cache && !row.cache_frontiers.empty() ? row.cache_frontiers.front() : 0});
        }
        auto result = engine_.score_branches(native, control);
        return ScoreBatch{std::move(result.logits), result.computed_tokens, result.reused_tokens,
                          result.prefill_ms,        result.forward_batches, result.largest_batch};
    };
    return evaluate(request, backend);
}

} // namespace ninfer::decision
