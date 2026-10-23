#pragma once

#include "models/common/frontend/output_session.h"
#include "models/common/frontend/resources.h"
#include "models/common/qwen/frontend/tool_contract.h"

#include <mutex>

namespace ninfer::text {
class GrammarCompiler;
}

namespace ninfer::models::qwen::frontend {

inline constexpr std::string_view kCanonicalReasoningCloseSerialization = "\n</think>\n\n";

struct PreparedOutput {
    std::shared_ptr<const ToolCallOutputContract> tools;
    bool starts_in_reasoning = false;
    std::optional<bool> enable_thinking;
    bool has_tool_history = false;
    std::string continuation_content;
};

class OutputResources {
public:
    OutputResources(const models::frontend::TextResources& resources,
                    std::size_t grammar_cache_bytes);
    ~OutputResources();
    text::GrammarCompiler& grammars() const;
    const std::shared_ptr<const text::Tokenizer> tokenizer;
    const StopPolicy defaults;
    std::shared_ptr<const models::frontend::ReasoningFormat> normal;
    std::shared_ptr<const models::frontend::ReasoningFormat> constrained;
private:
    const std::size_t cache_bytes_;
    mutable std::once_flag grammar_once_;
    mutable std::unique_ptr<text::GrammarCompiler> grammar_;
};

models::frontend::OutputSession create_output_session(
    const OutputResources& resources, const PreparedOutput& prompt, const StopPolicy& caller_stop,
    OutputOptions output, ThinkingControlOptions thinking,
    const std::optional<OutputConstraint>& constraint, const ToolChoice& tool_choice);

} // namespace ninfer::models::qwen::frontend
