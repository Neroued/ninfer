#pragma once

#include "text/grammar.h"

namespace ninfer::models::qwen::frontend {

std::unique_ptr<text::GrammarStateConstraint>
make_tool_name_constraint(std::shared_ptr<const text::GrammarVocabulary> vocabulary,
                          std::string_view reasoning_close, bool combined_body);

} // namespace ninfer::models::qwen::frontend
