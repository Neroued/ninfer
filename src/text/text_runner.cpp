#include "qus/text/text_runner.h"

#include "qus/core/nvtx_range.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace qus::text {

TextGenerationRunner::TextGenerationRunner(QwenTokenizer& tokenizer, qus::Engine& engine)
    : tokenizer_(tokenizer), engine_(engine) {}

TextGenerationResult TextGenerationRunner::generate(const std::vector<ChatMessage>& messages,
                                                     const TextGenerationOptions& options) {
    using Clock = std::chrono::steady_clock;

    if (options.max_new_tokens <= 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }

    const auto total_start = Clock::now();
    const auto render_start = Clock::now();
    const std::vector<int> stop_token_ids =
        resolve_stop_token_ids(tokenizer_, options.stop_token_ids);
    const std::string prompt              = render_qwen_chat(
        messages, ChatRenderOptions{.enable_thinking = options.enable_thinking});
    std::vector<int> prompt_token_ids     = tokenizer_.encode(prompt);
    const std::size_t required_context =
        prompt_token_ids.size() + static_cast<std::size_t>(options.max_new_tokens - 1);
    if (required_context > engine_.max_context()) {
        throw std::invalid_argument("prompt exceeds engine max_context");
    }
    const auto render_end = Clock::now();

    std::vector<int> generated_token_ids;
    generated_token_ids.reserve(static_cast<std::size_t>(options.max_new_tokens));

    const std::vector<int> decode_stop_token_ids = options.raw_output ? std::vector<int>{}
                                                                      : stop_token_ids;
    TokenStreamDecoder stream_decoder(
        tokenizer_, DecodeOptions{.skip_special_tokens = !options.raw_output,
                                  .stop_token_ids = decode_stop_token_ids});

    const auto emit_stream_text = [&](int token) {
        if (!options.stream_callback) { return; }
        const std::string text = stream_decoder.append(token);
        if (!text.empty()) { options.stream_callback(TextStreamChunk{.token_id = token, .text = text}); }
    };
    const auto is_stop = [&](int token) {
        return std::find(stop_token_ids.begin(), stop_token_ids.end(), token) !=
               stop_token_ids.end();
    };

    FinishReason finish_reason = FinishReason::Length;

    // If the caller already cancelled (e.g. the streaming client disconnected while
    // this request waited on the engine mutex), skip the expensive prefill entirely.
    const bool cancel_before_prefill = options.should_cancel && options.should_cancel();

    const auto prefill_start = Clock::now();
    int token = 0;
    if (!cancel_before_prefill) {
        const NvtxRange range("qus.prefill");
        token = engine_.prefill(prompt_token_ids);
    }
    const auto prefill_end = Clock::now();

    const auto decode_start = Clock::now();
    if (cancel_before_prefill) {
        finish_reason = FinishReason::Cancelled;
    } else {
        generated_token_ids.push_back(token);
        emit_stream_text(token);
        const NvtxRange range("qus.decode");
        if (is_stop(token)) {
            finish_reason = FinishReason::Stop;
        } else {
            while (static_cast<int>(generated_token_ids.size()) < options.max_new_tokens) {
                if (options.should_cancel && options.should_cancel()) {
                    finish_reason = FinishReason::Cancelled;
                    break;
                }
                token = engine_.decode_step();
                generated_token_ids.push_back(token);
                emit_stream_text(token);
                if (is_stop(token)) {
                    finish_reason = FinishReason::Stop;
                    break;
                }
            }
        }
    }
    const auto decode_end = Clock::now();

    if (options.stream_callback) {
        const std::string suffix = stream_decoder.finish();
        if (!suffix.empty()) {
            options.stream_callback(
                TextStreamChunk{.token_id = generated_token_ids.empty() ? -1
                                                                        : generated_token_ids.back(),
                                .text = suffix});
        }
    }

    std::string text = tokenizer_.decode(
        generated_token_ids,
        DecodeOptions{.skip_special_tokens = !options.raw_output,
                      .stop_token_ids = decode_stop_token_ids});
    const auto total_end = Clock::now();

    TextGenerationTimings timings;
    timings.render_tokenize_seconds = std::chrono::duration<double>(render_end - render_start).count();
    timings.prefill_seconds         = std::chrono::duration<double>(prefill_end - prefill_start).count();
    timings.decode_seconds          = std::chrono::duration<double>(decode_end - decode_start).count();
    timings.total_seconds           = std::chrono::duration<double>(total_end - total_start).count();

    return TextGenerationResult{.prompt_token_ids = std::move(prompt_token_ids),
                                .generated_token_ids = std::move(generated_token_ids),
                                .text = std::move(text),
                                .timings = timings,
                                .finish_reason = finish_reason};
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
