#include "qus/text/text_runner.h"

#include "qus/core/nvtx_range.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace qus::text {
namespace {

constexpr const char* kThinkClose = "</think>";

// Back a byte offset up to the start of a UTF-8 code point so a held-back tail
// never splits a multibyte character.
std::size_t backup_to_utf8_boundary(const std::string& buffer, std::size_t len) {
    if (len >= buffer.size()) { return buffer.size(); }
    while (len > 0 && (static_cast<unsigned char>(buffer[len]) & 0xC0) == 0x80) { --len; }
    return len;
}

// Splits a decoded token stream into a reasoning channel (the <think> block) and
// an answer channel. The opening <think> lives in the prompt, so generation
// starts inside reasoning whenever thinking is active; the closing "</think>"
// marks the switch to answer content, and leading whitespace after it (the
// template's blank lines) is dropped. Works for both incremental streaming (many
// push() calls) and one-shot splitting (single push() + flush()); both channels
// emit only on UTF-8 code-point boundaries.
class ThinkSplitter {
public:
    struct Result {
        std::string reasoning;
        std::string content;
    };

    explicit ThinkSplitter(bool starts_in_reasoning) : in_reasoning_(starts_in_reasoning) {}

    Result push(const std::string& text) {
        Result out;
        if (!in_reasoning_) {
            out.content = take_content(text);
            return out;
        }
        buffer_ += text;
        const std::size_t pos = buffer_.find(kThinkClose);
        if (pos != std::string::npos) {
            out.reasoning    = buffer_.substr(0, pos);
            std::string rest = buffer_.substr(pos + std::char_traits<char>::length(kThinkClose));
            buffer_.clear();
            in_reasoning_  = false;
            strip_leading_ = true;
            out.content    = take_content(rest);
            return out;
        }
        const std::size_t hold =
            std::min(buffer_.size(), std::char_traits<char>::length(kThinkClose) - 1);
        const std::size_t emit_len = backup_to_utf8_boundary(buffer_, buffer_.size() - hold);
        out.reasoning              = buffer_.substr(0, emit_len);
        buffer_.erase(0, emit_len);
        return out;
    }

    Result flush() {
        Result out;
        if (in_reasoning_) {
            out.reasoning = buffer_;
        } else {
            out.content = take_content(buffer_);
        }
        buffer_.clear();
        return out;
    }

private:
    std::string take_content(std::string piece) {
        if (strip_leading_) {
            std::size_t i = 0;
            while (i < piece.size() && std::isspace(static_cast<unsigned char>(piece[i])) != 0) {
                ++i;
            }
            piece.erase(0, i);
            if (!piece.empty()) { strip_leading_ = false; }
        }
        return piece;
    }

    bool in_reasoning_;
    bool strip_leading_ = false;
    std::string buffer_;
};

} // namespace

TextGenerationRunner::TextGenerationRunner(QwenTokenizer& tokenizer, qus::Engine& engine)
    : tokenizer_(tokenizer), engine_(engine) {}

TextGenerationResult TextGenerationRunner::generate(const std::vector<ChatMessage>& messages,
                                                    const TextGenerationOptions& options) {
    using Clock = std::chrono::steady_clock;

    const auto render_start          = Clock::now();
    ChatRenderOptions render_options = options.render_options;
    render_options.enable_thinking   = options.enable_thinking;
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const ChatMessage& message) { return message.has_media(); });
    if (has_media) {
        qus::model::Processor processor(tokenizer_);
        qus::model::ProcessedInput input = processor.process(messages, render_options);
        const std::uint32_t opener = generation_prompt_opener_tokens(tokenizer_, render_options);
        const std::uint32_t content_boundary =
            opener <= input.input_ids.size()
                ? static_cast<std::uint32_t>(input.input_ids.size()) - opener
                : static_cast<std::uint32_t>(input.input_ids.size());
        const double elapsed = std::chrono::duration<double>(Clock::now() - render_start).count();
        return run_tokens(input.input_ids, options, elapsed, content_boundary, &input);
    }

    const std::string prompt          = render_qwen_chat(messages, render_options);
    std::vector<int> prompt_token_ids = tokenizer_.encode(prompt);
    // Assistant-content boundary for cross-turn prefix reuse: the position right after the final
    // `<|im_start|>assistant\n` header, i.e. before the generation-prompt opener the template just
    // appended. A later thinking-stripped re-render diverges exactly here.
    const std::uint32_t opener = generation_prompt_opener_tokens(tokenizer_, render_options);
    const std::uint32_t content_boundary =
        opener <= prompt_token_ids.size()
            ? static_cast<std::uint32_t>(prompt_token_ids.size()) - opener
            : static_cast<std::uint32_t>(prompt_token_ids.size());
    const double render_tokenize_seconds =
        std::chrono::duration<double>(Clock::now() - render_start).count();
    return run_tokens(std::move(prompt_token_ids), options, render_tokenize_seconds,
                      content_boundary);
}

TextGenerationResult TextGenerationRunner::generate(std::span<const int> prompt_token_ids,
                                                    const TextGenerationOptions& options,
                                                    std::uint32_t content_boundary) {
    return run_tokens(std::vector<int>(prompt_token_ids.begin(), prompt_token_ids.end()), options,
                      0.0, content_boundary);
}

TextGenerationResult TextGenerationRunner::generate(qus::model::ProcessedInput& input,
                                                    const TextGenerationOptions& options,
                                                    std::uint32_t content_boundary,
                                                    std::function<void()> on_prefill_complete) {
    return run_tokens(input.input_ids, options, 0.0, content_boundary, &input,
                      std::move(on_prefill_complete));
}

TextGenerationResult TextGenerationRunner::run_tokens(std::vector<int> prompt_token_ids,
                                                      const TextGenerationOptions& options,
                                                      double render_tokenize_seconds,
                                                      std::uint32_t content_boundary,
                                                      qus::model::ProcessedInput* multimodal,
                                                      std::function<void()> on_prefill_complete) {
    using Clock = std::chrono::steady_clock;

    if (options.max_new_tokens <= 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }

    const auto total_start = Clock::now();
    const std::vector<int> stop_token_ids =
        resolve_stop_token_ids(tokenizer_, options.stop_token_ids);
    const std::size_t required_context =
        prompt_token_ids.size() + static_cast<std::size_t>(options.max_new_tokens - 1);
    if (required_context > engine_.max_context()) {
        throw std::invalid_argument("prompt exceeds engine max_context");
    }

    std::vector<int> generated_token_ids;
    generated_token_ids.reserve(static_cast<std::size_t>(options.max_new_tokens));

    const std::vector<int> decode_stop_token_ids =
        options.raw_output ? std::vector<int>{} : stop_token_ids;
    const bool skip_special_tokens = !options.raw_output && !options.preserve_special_tokens;
    TokenStreamDecoder stream_decoder(tokenizer_,
                                      DecodeOptions{.skip_special_tokens = skip_special_tokens,
                                                    .stop_token_ids      = decode_stop_token_ids});

    // Thinking output opens inside a <think> block that the prompt injected; split
    // the reasoning off the answer so callers see clean channels. Raw output is
    // passed through verbatim (tags and all).
    const bool split_thinking = options.enable_thinking && !options.raw_output;
    ThinkSplitter stream_splitter(split_thinking);

    const auto emit_split = [&](const ThinkSplitter::Result& split) {
        if (!options.stream_callback) { return; }
        if (!split.reasoning.empty()) {
            options.stream_callback(
                TextStreamChunk{.text = split.reasoning, .channel = TextChannel::Reasoning});
        }
        if (!split.content.empty()) {
            options.stream_callback(
                TextStreamChunk{.text = split.content, .channel = TextChannel::Content});
        }
    };
    const auto emit_stream_text = [&](int token) {
        if (!options.stream_callback) { return; }
        const std::string text = stream_decoder.append(token);
        if (!text.empty()) { emit_split(stream_splitter.push(text)); }
    };
    const auto is_stop = [&](int token) {
        return std::find(stop_token_ids.begin(), stop_token_ids.end(), token) !=
               stop_token_ids.end();
    };

    FinishReason finish_reason = FinishReason::Length;

    // If the caller already cancelled (e.g. the streaming client disconnected while
    // this request waited on the engine mutex), skip the expensive prefill entirely.
    const bool cancel_before_prefill = options.should_cancel && options.should_cancel();

    const auto prefill_start  = Clock::now();
    int token                 = 0;
    const auto finish_prefill = [&] {
        if (multimodal != nullptr) { std::vector<float>().swap(multimodal->patches); }
        if (on_prefill_complete) {
            std::function<void()> callback = std::move(on_prefill_complete);
            callback();
        }
    };
    try {
        if (!cancel_before_prefill) {
            // Apply this request's sampler before prefill so the config is resident in
            // device memory when the (graph-captured) decode kernels replay. Every
            // request sets it explicitly, so config never leaks between requests on a
            // shared engine; the default is greedy (temperature 0 == exact argmax).
            engine_.set_sampling(options.sampling);
            const NvtxRange range("qus.prefill");
            // Engine-level prefix caching: reuse the resident KV + GDN state when this request
            // extends the previous turn's token sequence, prefilling only the new suffix.
            token = multimodal != nullptr
                        ? engine_.prefill(*multimodal)
                        : engine_.prefill_cached(prompt_token_ids, content_boundary);
        }
        finish_prefill();
    } catch (...) {
        finish_prefill();
        throw;
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
        if (!suffix.empty()) { emit_split(stream_splitter.push(suffix)); }
        emit_split(stream_splitter.flush());
    }

    // Split the full decoded output once for the result fields (authoritative and
    // independent of whether the caller streamed).
    const std::string decoded = tokenizer_.decode(
        generated_token_ids, DecodeOptions{.skip_special_tokens = skip_special_tokens,
                                           .stop_token_ids      = decode_stop_token_ids});
    ThinkSplitter result_splitter(split_thinking);
    ThinkSplitter::Result head = result_splitter.push(decoded);
    ThinkSplitter::Result tail = result_splitter.flush();
    std::string reasoning      = std::move(head.reasoning) + tail.reasoning;
    std::string text           = std::move(head.content) + tail.content;
    const auto total_end       = Clock::now();

    TextGenerationTimings timings;
    timings.render_tokenize_seconds = render_tokenize_seconds;
    timings.prefill_seconds = std::chrono::duration<double>(prefill_end - prefill_start).count();
    timings.decode_seconds  = std::chrono::duration<double>(decode_end - decode_start).count();
    timings.total_seconds =
        render_tokenize_seconds + std::chrono::duration<double>(total_end - total_start).count();

    return TextGenerationResult{.prompt_token_ids    = std::move(prompt_token_ids),
                                .generated_token_ids = std::move(generated_token_ids),
                                .text                = std::move(text),
                                .reasoning           = std::move(reasoning),
                                .timings             = timings,
                                .finish_reason       = finish_reason};
}

std::uint32_t generation_prompt_opener_tokens(const QwenTokenizer& tokenizer,
                                              const ChatRenderOptions& options) {
    if (!options.add_generation_prompt) { return 0; }
    const std::string opener = options.enable_thinking ? "<think>\n" : "<think>\n\n</think>\n\n";
    return static_cast<std::uint32_t>(tokenizer.encode(opener).size());
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
