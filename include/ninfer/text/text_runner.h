#pragma once

#include "ninfer/runtime/engine.h"
#include "ninfer/model/processor.h"
#include "ninfer/text/chat_template.h"
#include "ninfer/text/tokenizer.h"

#include <functional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::text {

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
    double prefill_seconds         = 0.0;
    double decode_seconds          = 0.0;
    double total_seconds           = 0.0;
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
    bool raw_output    = false;
    // Default thinking-ON, matching the Qwen3.6 template's default generation prompt.
    bool enable_thinking         = true;
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
    ninfer::kernels::SamplingConfig sampling;
};

struct TextGenerationResult {
    std::vector<int> prompt_token_ids;
    std::vector<int> generated_token_ids;
    std::string text;      // answer only; the <think> block is split into `reasoning`
    std::string reasoning; // <think> block content, empty unless thinking is active
    TextGenerationTimings timings;
    FinishReason finish_reason = FinishReason::Length;
};

class TextGenerationRunner {
public:
    TextGenerationRunner(QwenTokenizer& tokenizer, ninfer::Engine& engine);

    // Prepares text or structured media messages, then prefills/decodes. Used by the CLI.
    TextGenerationResult generate(const std::vector<ChatMessage>& messages,
                                  const TextGenerationOptions& options);

    // Prefills/decodes an already-tokenized prompt, skipping render+encode. The
    // serving layer renders once in GenerationService::prepare() and reuses the
    // token ids here so a served request renders exactly once. `content_boundary`
    // is the absolute assistant-content boundary (prompt_tokens - opener length),
    // threaded into the engine prefix cache for cross-turn GDN reuse.
    TextGenerationResult generate(std::span<const int> prompt_token_ids,
                                  const TextGenerationOptions& options,
                                  std::uint32_t content_boundary);

    TextGenerationResult generate(ninfer::model::ProcessedInput& input,
                                  const TextGenerationOptions& options,
                                  std::uint32_t content_boundary,
                                  std::function<void()> on_prefill_complete = {});

private:
    // Shared prefill/decode body. render_tokenize_seconds is folded into the
    // total time so the metric stays meaningful regardless of the entry point.
    // `content_boundary` is passed straight to Engine::prefill_cached.
    TextGenerationResult run_tokens(std::vector<int> prompt_token_ids,
                                    const TextGenerationOptions& options,
                                    double render_tokenize_seconds, std::uint32_t content_boundary,
                                    ninfer::model::ProcessedInput* multimodal    = nullptr,
                                    std::function<void()> on_prefill_complete = {});

    QwenTokenizer& tokenizer_;
    ninfer::Engine& engine_;
};

std::vector<int> resolve_stop_token_ids(const QwenTokenizer& tokenizer,
                                        const std::vector<int>& overrides);

// Token length of the trailing generation-prompt opener that render_qwen_chat appends after the
// final `<|im_start|>assistant\n` header (`<think>\n` with thinking on, `<think>\n\n</think>\n\n`
// off). `<think>`/`</think>` are atomic special tokens, so the opener tokenizes independently of
// its left context and this length is a clean suffix split. Returns 0 when no generation prompt is
// added. Used to derive the assistant-content boundary for cross-turn prefix reuse.
std::uint32_t generation_prompt_opener_tokens(const QwenTokenizer& tokenizer,
                                              const ChatRenderOptions& options);

} // namespace ninfer::text
