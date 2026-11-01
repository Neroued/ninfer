#pragma once
#include "models/glm5next/frontend/tool_contract.h"
#include "text/grammar.h"

namespace ninfer::models::glm5next {
std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler,
                     std::shared_ptr<const ToolCallOutputContract> contract,
                     std::string_view reasoning_close, std::string_view continuation,
                     const std::optional<OutputConstraint>& body = {});
} // namespace ninfer::models::glm5next
