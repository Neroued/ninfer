#pragma once

#include "models/common/qwen/frontend/tool_contract.h"
#include "text/grammar.h"

namespace ninfer::models::qwen::frontend {
[[nodiscard]] std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const ToolCallOutputContract& contract,
                     std::string_view reasoning_close, std::string_view continuation,
                     const std::optional<OutputConstraint>& body = {});
} // namespace ninfer::models::qwen::frontend
