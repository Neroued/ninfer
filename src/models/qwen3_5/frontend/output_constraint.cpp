#include "models/qwen3_5/frontend/output_constraint.h"

#include "models/qwen3_5/frontend/chat_template.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/tokenizer.h"
#include "text/grammar.h"

#include <string>
#include <string_view>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {

ConstraintCompiler::ConstraintCompiler(std::shared_ptr<const Tokenizer> tokenizer,
                                       std::vector<TokenId> eos, std::size_t cache_bytes,
                                       const OutputConstraintLimits& limits)
    : tokenizer_(std::move(tokenizer)), eos_(std::move(eos)), cache_bytes_(cache_bytes),
      limits_(limits) {}

ConstraintCompiler::~ConstraintCompiler() = default;

text::GrammarCompiler& ConstraintCompiler::grammars() const {
    std::call_once(once_, [&] {
        // Upstream 08902f73 Frontend::Impl::grammars(): the decoded bytes of every ordinary
        // token; special and invalid IDs stay empty (forbidden), EOS is passed separately and is
        // licensed only where the language is complete. The tags <think>, </think>,
        // <tool_call>... are ordinary tokens in every Qwen3.8 artifact (special=false), so the
        // grammar can generate them as single tokens or split into byte-level pieces alike.
        std::vector<std::string> vocabulary(tokenizer_->vocab_size());
        for (std::size_t id = 0; id < vocabulary.size(); ++id) {
            const auto token = static_cast<TokenId>(id);
            if (!tokenizer_->is_valid_token(token)) { continue; }
            const DecodedTokenView decoded = tokenizer_->decoded_token(token);
            if (!decoded.special) { vocabulary[id] = std::string(decoded.bytes); }
        }
        compiler_ = std::make_unique<text::GrammarCompiler>(std::move(vocabulary), eos_,
                                                            cache_bytes_, limits_);
    });
    return *compiler_;
}

void validate_tool_choice(const ToolChoice& tool_choice) {
    if (tool_choice.mode == ToolChoiceMode::Required || tool_choice.allowed_names ||
        !tool_choice.parallel || tool_choice.constraints != ToolConstraintMode::Automatic) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint,
                           "tool_choice selection, cardinality and strict tool constraints are not "
                           "supported by this engine yet",
                           {}, RequestErrorSource::Tools);
    }
}

std::unique_ptr<text::GrammarSession> make_constraint_session(const ConstraintCompiler& compiler,
                                                              const PreparedPromptData& prompt,
                                                              const OutputConstraint& constraint,
                                                              const StopPolicy& caller_stop,
                                                              const OutputOptions& output) {
    if (prompt.has_active_tools) {
        throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                           "output constraints cannot be combined with active tools yet");
    }
    if (prompt.continuation) {
        // D7: an assistant continuation would have to start the matcher inside the rendered
        // prefix (upstream continuation_content); not supported in this stage.
        throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                           "output constraints cannot continue an assistant message");
    }
    if (!caller_stop.strings.empty() || !caller_stop.token_ids.empty() ||
        !caller_stop.include_model_defaults || caller_stop.publish_stop_token) {
        throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                           "output constraints end at the model EOS and cannot use custom stops",
                           "stop");
    }
    if (output.raw || output.preserve_special_tokens) {
        throw RequestError(RequestErrorKind::UnsupportedOutputCombination,
                           "output constraints require text output without raw or special tokens");
    }
    const std::string_view close =
        prompt.starts_in_reasoning ? kCanonicalReasoningCloseSerialization : std::string_view{};
    try {
        return compiler.grammars().compile(constraint, close, {});
    } catch (const RequestError&) { throw; } catch (const std::invalid_argument& error) {
        throw RequestError(text::constraint_error_kind(constraint.kind), error.what());
    }
}

} // namespace ninfer::models::qwen3_5::frontend
