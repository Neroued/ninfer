#pragma once

#include "models/common/frontend/output_resources.h"
#include "models/common/frontend/output_session.h"
#include "models/inkling2/frontend/prompt_layout.h"
#include "models/inkling2/frontend/resources.h"
#include "models/inkling2/frontend/tool_contract.h"

#include <memory>
#include <optional>
#include <string>

namespace ninfer::models::inkling2 {

struct PreparedOutput {
    std::shared_ptr<const ToolCallOutputContract> tools;
    InitialBlock initial_block = InitialBlock::BetweenBlocks;
    bool thinking_enabled      = true;
    std::string continuation_content;
};

class OutputResources {
public:
    OutputResources(const FrontendResources& resources, std::size_t grammar_cache_bytes);

    models::frontend::OutputResources text;
    FramingTokens framing;
    std::shared_ptr<const models::frontend::ReasoningFormat> to_text;
    std::shared_ptr<const models::frontend::ReasoningFormat> to_header;
};

models::frontend::OutputSession
create_output_session(const OutputResources& resources, const PreparedOutput& prompt,
                      const StopPolicy& stop, OutputOptions output, ThinkingControlOptions thinking,
                      const std::optional<OutputConstraint>& constraint, const ToolChoice& choice);

} // namespace ninfer::models::inkling2
