#pragma once

#include "models/common/qwen/frontend/tool_contract.h"
#include "models/common/qwen/frontend/tool_output_error.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen::frontend {

struct ParsedToolCallOutput {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<GeneratedToolCall> tool_calls;
    ToolCallParseDiagnostics diagnostics;
};

[[nodiscard]] ParsedToolCallOutput
parse_qwen_tool_call_output(const std::string& text, std::size_t max_tool_name_length,
                            const ToolCallOutputContract& contract);

// Consumes committed model text exactly once. A recognized tool header commits the interpretation;
// subsequent model-format errors are request-local failures, never a retraction into ordinary text.
// Arguments and aggregate calls share the same append-only JSON encoding.
class ToolCallOutputDecoder {
public:
    ToolCallOutputDecoder(std::shared_ptr<const ToolCallOutputContract> contract,
                          std::size_t max_tool_name_length);
    ~ToolCallOutputDecoder();
    ToolCallOutputDecoder(ToolCallOutputDecoder&&) noexcept;
    ToolCallOutputDecoder& operator=(ToolCallOutputDecoder&&) noexcept;

    [[nodiscard]] bool in_tool_region() const noexcept;
    [[nodiscard]] std::vector<OutputEvent> feed(std::string_view text);
    void initialize_continuation(std::string_view prefix);
    [[nodiscard]] std::vector<OutputEvent> finish(FinishReason reason = FinishReason::StopToken);
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept;
    [[nodiscard]] ToolCallParseDiagnostics diagnostics() const noexcept;
    [[nodiscard]] std::optional<ToolOutputError> error() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::models::qwen::frontend
