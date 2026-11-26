#pragma once

#include "models/inkling2/frontend/prompt_layout.h"
#include "models/inkling2/frontend/resources.h"
#include "models/inkling2/frontend/tool_contract.h"
#include "ninfer/types.h"
#include "text/grammar.h"

#include <memory>
#include <optional>
#include <string_view>

namespace ninfer::models::inkling2 {

[[nodiscard]] std::unique_ptr<text::GrammarSession>
compile_inkling_grammar(text::GrammarCompiler& compiler, const FramingTokens& framing,
                        InitialBlock initial_block, std::string_view continuation,
                        const std::optional<OutputConstraint>& body,
                        const std::shared_ptr<const ToolCallOutputContract>& tools,
                        bool allow_thinking, bool thinking_budget_active);

} // namespace ninfer::models::inkling2
