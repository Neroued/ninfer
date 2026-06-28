#pragma once

#include "qus/runtime/engine.h"
#include "qus/text/chat_template.h"
#include "qus/text/tokenizer.h"

#include <string>
#include <vector>

namespace qus::text {

struct TextGenerationOptions {
    int max_new_tokens = 128;
    bool raw_output = false;
    bool print_token_ids = false;
    std::vector<int> stop_token_ids;
};

struct TextGenerationResult {
    std::vector<int> prompt_token_ids;
    std::vector<int> generated_token_ids;
    std::string text;
};

class TextGenerationRunner {
public:
    TextGenerationRunner(QwenTokenizer& tokenizer, qus::Engine& engine);

    TextGenerationResult generate(const std::vector<ChatMessage>& messages,
                                  const TextGenerationOptions& options);

private:
    QwenTokenizer& tokenizer_;
    qus::Engine& engine_;
};

std::vector<int> resolve_stop_token_ids(const QwenTokenizer& tokenizer,
                                        const std::vector<int>& overrides);

} // namespace qus::text
