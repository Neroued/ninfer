#pragma once

#include "qus/runtime/engine.h"
#include "qus/text/chat_template.h"
#include "qus/text/tokenizer.h"

#include <functional>
#include <string>
#include <vector>

namespace qus::text {

struct TextStreamChunk {
    int token_id = 0;
    std::string text;
};

using TextStreamCallback = std::function<void(const TextStreamChunk&)>;

struct TextGenerationTimings {
    double render_tokenize_seconds = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    double total_seconds = 0.0;
};

// Why generation stopped. Cancelled means an external caller (e.g. a disconnected
// streaming client) asked to stop early via should_cancel; it is not an error.
enum class FinishReason {
    Stop,
    Length,
    Cancelled,
};

struct TextGenerationOptions {
    int max_new_tokens = 128;
    bool raw_output = false;
    bool enable_thinking = false;
    std::vector<int> stop_token_ids;
    TextStreamCallback stream_callback;
    // Checked before each decode step; return true to stop early. Used by the
    // serving layer to abort generation when the HTTP client disconnects.
    std::function<bool()> should_cancel;
};

struct TextGenerationResult {
    std::vector<int> prompt_token_ids;
    std::vector<int> generated_token_ids;
    std::string text;
    TextGenerationTimings timings;
    FinishReason finish_reason = FinishReason::Length;
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
