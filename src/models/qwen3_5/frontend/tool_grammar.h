#pragma once

#include "models/qwen3_5/frontend/tool_contract.h"
#include "text/grammar.h"

namespace ninfer::models::qwen3_5::frontend {
[[nodiscard]] std::unique_ptr<text::GrammarSession>
compile_tool_grammar(text::GrammarCompiler& compiler, const ToolCallOutputContract& contract,
                     std::string_view reasoning_close, std::string_view continuation,
                     const std::optional<OutputConstraint>& body = {});
} // namespace ninfer::models::qwen3_5::frontend
