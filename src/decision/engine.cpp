#include "decision/decision.h"
#include "ninfer/engine.h"

#include <algorithm>
#include <chrono>
#include <utility>

namespace ninfer::decision {

Json DecisionEngine::run(const Request& request, const PreparationControl& control,
                         std::span<const OwnedMedia> images) {
    if (!images.empty() && request.contexts.size() != 1)
        throw std::invalid_argument("image decisions require exactly one state");
    BranchScorePrompt picture;
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
        for (auto image = images.rbegin(); image != images.rend(); ++image)
            input.messages[1].parts.insert(input.messages[1].parts.begin(),
                                           {.kind = MessagePartKind::Media, .media = *image});
        auto preparation     = control;
        const auto deadline  = std::chrono::steady_clock::now() +
                               std::chrono::milliseconds(engine_.options().pending_timeout_ms);
        preparation.deadline = control.deadline == std::chrono::steady_clock::time_point{}
                                   ? deadline
                                   : std::min(deadline, control.deadline);
        if (!images.empty() && picture.tokens.empty()) {
            picture = engine_.prepare_branch_prompt(std::move(input), preparation);
            return picture.tokens;
        }
        return engine_.tokenize_prompt(std::move(input), preparation);
    };
    backend.score = [&](std::vector<ScoreRow> rows, bool reuse_prefix) {
        check();
        std::vector<BranchScoreRow> native;
        native.reserve(rows.size());
        for (const auto& row : rows) {
            native.push_back({row.prefix, row.suffix, row.candidates,
                              !row.cache_frontiers.empty() ? row.cache_frontiers.back() : 0,
                              !row.cache_frontiers.empty() ? row.cache_frontiers.front() : 0});
        }
        auto result = engine_.score_branches(native, control, reuse_prefix,
                                             images.empty() ? nullptr : &picture);
        return ScoreBatch{std::move(result.logits), result.computed_tokens, result.reused_tokens,
                          result.prefill_ms,        result.forward_batches, result.largest_batch};
    };
    return evaluate(request, backend);
}

} // namespace ninfer::decision
