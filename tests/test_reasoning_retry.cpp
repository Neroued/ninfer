// plan_reasoning_retry (serve --reasoning-close-retry): when a reply is continued, with which
// tokens, and when it is left alone.
#include "serve/reasoning_retry.h"

#include <iostream>
#include <vector>

namespace {

using ninfer::FinishReason;
using ninfer::TokenId;
using ninfer::serve::plan_reasoning_retry;
using ninfer::serve::ReasoningRetryInput;
using ninfer::serve::ReasoningRetryTrigger;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << "\n";
    return 1;
}

// Stop and </think> ids as in the Qwen3.8 tokenizer; the other ids are illustrative.
ninfer::ReasoningGuardTokens qwen_tokens() {
    ninfer::ReasoningGuardTokens tokens;
    tokens.close_suffix   = {198, 248069, 271};
    tokens.stop_token_ids = {248046, 248044};
    tokens.think_close    = {248069};
    return tokens;
}

ReasoningRetryInput stopped(const std::vector<TokenId>& generated) {
    return ReasoningRetryInput{.starts_in_reasoning      = true,
                               .finish_reason            = FinishReason::StopToken,
                               .generated                = generated,
                               .content                  = "",
                               .tool_calls               = 0,
                               .thinking_control_applied = false,
                               .requested_output_tokens  = 32000};
}

} // namespace

int main() {
    int failures      = 0;
    const auto tokens = qwen_tokens();

    // The banco-agentico case: the first token after "<think>\n" is <|im_end|>.
    {
        const std::vector<TokenId> generated{248046};
        const auto plan = plan_reasoning_retry(stopped(generated), tokens);
        failures += check(plan.trigger == ReasoningRetryTrigger::StopInReasoning &&
                              plan.appended == tokens.close_suffix && plan.kept_tokens == 0 &&
                              plan.injected_tokens == 3 && plan.remaining_output_tokens == 31997U,
                          "a first-token stop is not closed and continued with the close suffix");
    }
    // Stop after some reasoning (B4): the reasoning is kept, then the close.
    {
        const std::vector<TokenId> generated{100, 101, 102, 248044};
        const auto plan = plan_reasoning_retry(stopped(generated), tokens);
        failures +=
            check(plan.trigger == ReasoningRetryTrigger::StopInReasoning &&
                      plan.appended == std::vector<TokenId>({100, 101, 102, 198, 248069, 271}) &&
                      plan.kept_tokens == 3,
                  "a stop inside the reasoning does not keep the reasoning before the close");
    }
    // Stop right after "</think>" with blank content: resumed from the same point, nothing added.
    {
        const std::vector<TokenId> generated{100, 248069, 271, 248046};
        auto input      = stopped(generated);
        input.content   = "\n";
        const auto plan = plan_reasoning_retry(input, tokens);
        failures += check(plan.trigger == ReasoningRetryTrigger::EmptyAfterClose &&
                              plan.appended == std::vector<TokenId>({100, 248069, 271}) &&
                              plan.injected_tokens == 0,
                          "an empty reply after </think> is not resumed from the same point");
    }
    // Left alone: a normal answer, a tool call, a length finish, no thinking, a caller stop token,
    // an applied thinking budget, no room after the close.
    {
        const std::vector<TokenId> generated{100, 248069, 271, 300, 248046};
        auto answer    = stopped(generated);
        answer.content = "Done.";
        failures +=
            check(plan_reasoning_retry(answer, tokens).trigger == ReasoningRetryTrigger::None,
                  "a reply with content was retried");
        auto tool       = stopped(generated);
        tool.tool_calls = 1;
        failures += check(plan_reasoning_retry(tool, tokens).trigger == ReasoningRetryTrigger::None,
                          "a reply with a tool call was retried");
    }
    {
        const std::vector<TokenId> generated{100, 101};
        auto length          = stopped(generated);
        length.finish_reason = FinishReason::OutputLimit;
        failures +=
            check(plan_reasoning_retry(length, tokens).trigger == ReasoningRetryTrigger::None,
                  "a reply cut by the output limit was retried");
    }
    {
        const std::vector<TokenId> generated{248046};
        auto plain                = stopped(generated);
        plain.starts_in_reasoning = false;
        failures +=
            check(plan_reasoning_retry(plain, tokens).trigger == ReasoningRetryTrigger::None,
                  "a reply without thinking was retried");
        auto budget                     = stopped(generated);
        budget.thinking_control_applied = true;
        failures +=
            check(plan_reasoning_retry(budget, tokens).trigger == ReasoningRetryTrigger::None,
                  "a reply whose thinking budget was applied was retried");
        auto tight                    = stopped(generated);
        tight.requested_output_tokens = 3;
        failures +=
            check(plan_reasoning_retry(tight, tokens).trigger == ReasoningRetryTrigger::None,
                  "a retry was planned without room for one token after the close");
        auto open                    = stopped(generated);
        open.requested_output_tokens = 0;
        const auto plan              = plan_reasoning_retry(open, tokens);
        failures += check(plan.trigger == ReasoningRetryTrigger::StopInReasoning &&
                              !plan.remaining_output_tokens,
                          "an unbounded request did not leave the continuation budget unset");
    }
    {
        const std::vector<TokenId> generated{100, 777};
        failures += check(plan_reasoning_retry(stopped(generated), tokens).trigger ==
                              ReasoningRetryTrigger::None,
                          "a caller stop token was treated as the model's end of turn");
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
