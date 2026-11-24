#pragma once

#include "models/common/frontend/output_session.h"
#include "models/common/frontend/output_resources.h"
#include "models/common/qwen/frontend/tool_contract.h"

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
    models::frontend::OutputResources text;
    std::shared_ptr<const models::frontend::ReasoningFormat> normal;
    std::shared_ptr<const models::frontend::ReasoningFormat> constrained;
};

models::frontend::OutputSession create_output_session(
    const OutputResources& resources, const PreparedOutput& prompt, const StopPolicy& caller_stop,
    OutputOptions output, ThinkingControlOptions thinking,
    const std::optional<OutputConstraint>& constraint, const ToolChoice& tool_choice);

} // namespace ninfer::models::qwen::frontend
