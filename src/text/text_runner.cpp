#include "qus/text/text_runner.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace qus::text {

TextGenerationRunner::TextGenerationRunner(QwenTokenizer& tokenizer, qus::Engine& engine)
    : tokenizer_(tokenizer), engine_(engine) {}

TextGenerationResult TextGenerationRunner::generate(const std::vector<ChatMessage>& messages,
                                                     const TextGenerationOptions& options) {
    if (options.max_new_tokens <= 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }

    const std::vector<int> stop_token_ids =
        resolve_stop_token_ids(tokenizer_, options.stop_token_ids);
    const std::string prompt              = render_qwen_chat(messages);
    std::vector<int> prompt_token_ids     = tokenizer_.encode(prompt);
    if (prompt_token_ids.size() > engine_.max_context()) {
        throw std::invalid_argument("prompt exceeds engine max_context");
    }

    std::vector<int> generated_token_ids =
        engine_.generate(prompt_token_ids, options.max_new_tokens);
    std::string text = tokenizer_.decode(
        generated_token_ids,
        DecodeOptions{.skip_special_tokens = !options.raw_output, .stop_token_ids = stop_token_ids});

    return TextGenerationResult{.prompt_token_ids = std::move(prompt_token_ids),
                                .generated_token_ids = std::move(generated_token_ids),
                                .text = std::move(text)};
}

std::vector<int> resolve_stop_token_ids(const QwenTokenizer& tokenizer,
                                        const std::vector<int>& overrides) {
    if (overrides.empty()) { return tokenizer.default_stop_token_ids(); }

    std::vector<int> resolved;
    resolved.reserve(overrides.size());
    for (const int id : overrides) {
        if (id < 0) { throw std::invalid_argument("stop token id must be nonnegative"); }
        if (std::find(resolved.begin(), resolved.end(), id) == resolved.end()) {
            resolved.push_back(id);
        }
    }
    return resolved;
}

} // namespace qus::text
