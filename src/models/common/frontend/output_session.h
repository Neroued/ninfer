#pragma once

#include "ninfer/types.h"
#include "runtime/contract/request.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::text {
class GrammarSession;
class Tokenizer;
} // namespace ninfer::text

namespace ninfer::models::frontend {
using PublishedOutput = std::vector<OutputEvent>;

enum class ToolOutputErrorKind { InvalidModelOutput, ContractViolation };

struct ToolOutputError {
    ToolOutputErrorKind kind;
    std::string message;

    std::exception_ptr exception() const {
        if (kind == ToolOutputErrorKind::ContractViolation)
            return std::make_exception_ptr(
                std::logic_error("tool constraint/decoder disagreement: " + message));
        return std::make_exception_ptr(RequestError(RequestErrorKind::InvalidToolOutput, message,
                                                    {}, RequestErrorSource::Tools));
    }
};

// Immutable model-provided framing. Prefix failure data is shared across request previews.
struct ReasoningFormat {
    ReasoningFormat(std::string close, std::string execution_boundary,
                    std::shared_ptr<const std::vector<TokenId>> control_tokens,
                    bool strip_content_leading);
    std::string close;
    std::string execution_boundary;
    std::vector<std::size_t> boundary_failure;
    std::shared_ptr<const std::vector<TokenId>> control_tokens;
    bool strip_content_leading;
};

// Consumes committed content only; format-specific branches never enter Engine.
class BodyDecoder {
public:
    virtual ~BodyDecoder()                                            = default;
    virtual PublishedOutput feed(std::string_view bytes)              = 0;
    virtual PublishedOutput finish(FinishReason reason)               = 0;
    virtual std::vector<GeneratedToolCall> take_tool_calls() noexcept = 0;
    virtual ToolCallParseDiagnostics diagnostics() const noexcept     = 0;
    virtual std::optional<ToolOutputError> error() const              = 0;
    virtual ConstraintOutputBranch branch() const noexcept            = 0;
};

struct OutputSessionOptions {
    std::shared_ptr<const text::Tokenizer> tokenizer;
    std::shared_ptr<const ReasoningFormat> reasoning;
    StopPolicy stop;
    OutputOptions presentation;
    bool starts_in_reasoning = false;
    bool has_content_prefix  = false;
    ThinkingControlOptions thinking;
};

class OutputSession {
public:
    OutputSession() noexcept;
    OutputSession(OutputSessionOptions options, std::unique_ptr<BodyDecoder> body,
                  std::unique_ptr<text::GrammarSession> grammar);
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
    [[nodiscard]] std::optional<ToolOutputError> output_error() const;
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::frontend
