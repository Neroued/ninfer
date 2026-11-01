#pragma once

#include "models/common/frontend/output_resources.h"
#include "models/common/frontend/output_session.h"
#include "models/glm5next/frontend/tool_contract.h"

namespace ninfer::models::glm5next {

struct PreparedOutput {
    std::shared_ptr<const ToolCallOutputContract> tools;
    bool starts_in_reasoning = false;
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

models::frontend::OutputSession
create_output_session(const OutputResources& resources, const PreparedOutput& prompt,
                      const StopPolicy& stop, OutputOptions output, ThinkingControlOptions thinking,
                      const std::optional<OutputConstraint>& constraint, const ToolChoice& choice);

} // namespace ninfer::models::glm5next
