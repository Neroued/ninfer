#pragma once

#include "models/common/qwen/frontend/tool_output_error.h"
#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::text {
class GrammarSession;
}

namespace ninfer::models::qwen {
namespace frontend {
class Tokenizer;
struct ToolCallOutputContract;
} // namespace frontend
class Frontend;

using PublishedOutput = std::vector<OutputEvent>;

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview_model(std::span<const TokenId> tokens,
                                                        std::uint32_t total_budget_remaining,
                                                        FinishReason limit_reason);
    [[nodiscard]] std::uint32_t
    model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept;
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId> tokens,
                                                          std::uint32_t total_budget_remaining);
    void validate_generation_capacity(std::uint32_t effective_output_tokens) const;
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    void discard_preview();
    [[nodiscard]] bool constrained() const noexcept;
    void observe_constraint(bool timings, double prepare_seconds) noexcept;
    void constraint_uploaded(std::size_t bytes) noexcept;
    [[nodiscard]] std::optional<ConstraintObservation> constraint_observation() const;
    [[nodiscard]] std::uint32_t grammar_masks(std::span<const TokenId> drafts,
                                              std::span<std::uint32_t> words);
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] std::optional<frontend::ToolOutputError> output_error() const;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    class Impl;
    OutputSession(std::shared_ptr<const frontend::Tokenizer> tokenizer, StopPolicy policy,
                  OutputOptions output, bool starts_in_reasoning, ThinkingControlOptions thinking,
                  std::shared_ptr<const std::vector<TokenId>> thinking_control_tokens,
                  std::shared_ptr<const frontend::ToolCallOutputContract> tool_call_output,
                  std::unique_ptr<text::GrammarSession> grammar = {},
                  std::string_view continuation = {}, bool combined = false);
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

} // namespace ninfer::models::qwen
