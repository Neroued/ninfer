#pragma once

#include "ninfer/types.h"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace ninfer::runtime {

// A bounded tokenizer or partial media plan supplies a lower bound, not the full prompt length.
[[nodiscard]] inline std::string context_length_error_message(std::size_t tokens,
                                                              std::size_t max_context,
                                                              bool lower_bound = false) {
    return "maximum context length is " + std::to_string(max_context) +
           " tokens, but the prepared prompt has " + (lower_bound ? "at least " : "") +
           std::to_string(tokens) + " tokens";
}

// Engine has applied explicit overrides to the Frontend-selected preset and validated the result.
struct ResolvedExecutionOptions {
    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    ThinkingControlOptions thinking;
};

struct ResolvedRequestOptions {
    std::optional<OutputConstraint> constraint;
    ToolChoice tool_choice;
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

enum class ContinuationAction : std::uint8_t {
    Decode,
    ApplyTargetControl,
};

struct OutputDecision {
    std::uint32_t accepted_tokens   = 0;
    FinishReason finish_reason      = FinishReason::None;
    ContinuationAction continuation = ContinuationAction::Decode;
    // Empty or one token-aligned frontier within the accepted span where model-history
    // reconstruction gains an execution split. Frontend owns detection; Engine only transports
    // this relative position.
    std::optional<std::uint32_t> prefix_execution_split_after;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Non-owning cancellation observation used while the worker advances a context transaction. The
// request record owns the flag for longer than Program can retain this view.
struct CancellationFlagView {
    const std::atomic<bool>* flag = nullptr;

    [[nodiscard]] bool requested() const noexcept {
        return flag != nullptr && flag->load(std::memory_order_acquire);
    }
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    bool publish_continuation             = true;
};

} // namespace ninfer::runtime
