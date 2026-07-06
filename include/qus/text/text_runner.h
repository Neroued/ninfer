#pragma once

#include "qus/runtime/engine.h"
#include "qus/text/chat_template.h"
#include "qus/text/tokenizer.h"

#include <functional>
#include <string>
#include <vector>

namespace qus::text {

// The channel a streamed delta belongs to. When thinking is enabled the model's
// output opens inside a <think> block (the opening tag lives in the prompt); the
// runner routes everything up to "</think>" to Reasoning and the rest to Content.
enum class TextChannel {
    Content,
    Reasoning,
};

struct TextStreamChunk {
    std::string text;
    TextChannel channel = TextChannel::Content;
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
    bool preserve_special_tokens = false;
    ChatRenderOptions render_options;
    std::vector<int> stop_token_ids;
    TextStreamCallback stream_callback;
    // Checked before each decode step; return true to stop early. Used by the
    // serving layer to abort generation when the HTTP client disconnects.
    std::function<bool()> should_cancel;
    // Sampler config applied to the engine before prefill. Default is greedy
    // (temperature 0 => exact argmax), so callers that leave it untouched decode
    // deterministically; front-ends opt in to sampling by populating it.
    qus::kernels::SamplingConfig sampling;
};

struct TextGenerationResult {
    std::vector<int> prompt_token_ids;
    std::vector<int> generated_token_ids;
    std::string text;       // answer only; the <think> block is split into `reasoning`
    std::string reasoning;  // <think> block content, empty unless thinking is active
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
