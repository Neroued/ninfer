#pragma once

// Reasoning-close retry (--reasoning-close-retry, on in --profile flash-next). Qwen3.8-Flash-Next
// sometimes emits its end-of-turn token while the thinking phase is still open, at the very first
// token after "<think>\n" or after some reasoning: the reply comes back with empty content and no
// tool call, and an agent harness stops. When that happens the service closes the thinking with
// the canonical "\n</think>\n\n" and lets the model continue once, from the reply so far. A reply
// that stops right after "</think>" with blank content is resumed once from the same point with a
// different seed. Nothing changes for a reply that does not end this way: the first generation runs
// exactly as without the option.

#include "ninfer/types.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::serve {

enum class ReasoningRetryTrigger : std::uint8_t {
    None,
    StopInReasoning, // terminal token while thinking was open (no "</think>" in the reply)
    EmptyAfterClose, // terminal token after "</think>" with blank content and no tool call
};

[[nodiscard]] const char* reasoning_retry_trigger_name(ReasoningRetryTrigger trigger) noexcept;

struct ReasoningRetryInput {
    bool starts_in_reasoning   = false;
    FinishReason finish_reason = FinishReason::None;
    std::span<const TokenId> generated; // the reply's token ids, terminal token included
    std::string_view content;
    std::size_t tool_calls        = 0;
    bool thinking_control_applied = false;
    // Request output budget (0 = the server default, not bounded here).
    std::uint32_t requested_output_tokens = 0;
};

struct ReasoningRetryPlan {
    ReasoningRetryTrigger trigger = ReasoningRetryTrigger::None;
    // Tokens appended to the chat prompt: the reply without its terminal token, then the close
    // suffix for StopInReasoning.
    std::vector<TokenId> appended;
    std::uint32_t kept_tokens     = 0; // reply tokens kept (terminal token dropped)
    std::uint32_t injected_tokens = 0; // close-suffix tokens
    // Output budget left for the continuation when the request set one.
    std::optional<std::uint32_t> remaining_output_tokens;
};

[[nodiscard]] ReasoningRetryPlan plan_reasoning_retry(const ReasoningRetryInput& input,
                                                      const ReasoningGuardTokens& tokens);

} // namespace ninfer::serve
