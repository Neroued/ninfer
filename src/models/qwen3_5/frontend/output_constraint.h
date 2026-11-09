#pragma once

// Structured output (constrained decoding) on the Qwen frontend: the per-model grammar compiler
// and the per-request matcher construction. capy-flash file (structured output P3); the matcher
// itself is the text library ported from Neroued/ninfer 77c2e5e2 (src/text/grammar.h), the
// vocabulary and admission rules follow upstream 08902f73 (frontend.cpp make_output_session).

#include "ninfer/types.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

namespace ninfer::text {
class GrammarCompiler;
class GrammarSession;
} // namespace ninfer::text

namespace ninfer::models::qwen3_5 {
struct PreparedPromptData;

namespace frontend {
class Tokenizer;

// One per model: the XGrammar vocabulary and compiler cache are built on the first constrained
// request, in the submitting thread, so an unconstrained server never pays for them.
class ConstraintCompiler {
public:
    ConstraintCompiler(std::shared_ptr<const Tokenizer> tokenizer, std::vector<TokenId> eos,
                       std::size_t cache_bytes, const OutputConstraintLimits& limits = {});
    ~ConstraintCompiler();
    ConstraintCompiler(const ConstraintCompiler&)            = delete;
    ConstraintCompiler& operator=(const ConstraintCompiler&) = delete;

    [[nodiscard]] text::GrammarCompiler& grammars() const;

private:
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::vector<TokenId> eos_;
    std::size_t cache_bytes_ = 0;
    OutputConstraintLimits limits_;
    mutable std::once_flag once_;
    mutable std::unique_ptr<text::GrammarCompiler> compiler_;
};

// Engine-level tool selection beyond today's Auto/None is not part of this build (structured
// output stage B). Throws RequestError for any selection it would otherwise ignore.
void validate_tool_choice(const ToolChoice& tool_choice);

// Validates the request combination and compiles its matcher (RequestError on rejection). A
// prompt that starts in thinking prefixes the language with "anything until the canonical
// close \n</think>\n\n", so the thinking channel stays free and only the content is constrained.
[[nodiscard]] std::unique_ptr<text::GrammarSession>
make_constraint_session(const ConstraintCompiler& compiler, const PreparedPromptData& prompt,
                        const OutputConstraint& constraint, const StopPolicy& caller_stop,
                        const OutputOptions& output);

} // namespace frontend
} // namespace ninfer::models::qwen3_5
