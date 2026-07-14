#include "ninfer/serve/generation_service.h"

#include "ninfer/serve/tool_call_parser.h"
#include "ninfer/serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <utility>

namespace ninfer::serve {
namespace {

[[noreturn]] void throw_processor_error(const ninfer::model::ProcessorError& exception) {
    ApiError error;
    error.type    = "invalid_request_error";
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::model::ProcessorErrorKind::BudgetExceeded:
        error.status = 413;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::model::ProcessorErrorKind::RemoteUnavailable:
        error.status = 502;
        error.code   = "media_fetch_failed";
        break;
    case ninfer::model::ProcessorErrorKind::RemoteTimeout:
        error.status = 504;
        error.code   = "media_fetch_timeout";
        break;
    }
    throw ApiException(std::move(error));
}

// Back a byte offset up to the start of a UTF-8 code point so a held-back tail
// never splits a multibyte character. `len` is clamped to [0, buffer.size()].
std::size_t backup_to_utf8_boundary(const std::string& buffer, std::size_t len) {
    if (len >= buffer.size()) { return buffer.size(); }
    while (len > 0 && (static_cast<unsigned char>(buffer[len]) & 0xC0) == 0x80) { --len; }
    return len;
}

// Detokenized stop-string matching. Emitted text never includes the stop string.
// A tail of up to (max_len - 1) bytes is held back so a stop string spanning two
// deltas is still caught before its prefix is emitted.
class StopSequences {
public:
    struct Result {
        std::string emit;
        bool stopped = false;
    };

    explicit StopSequences(const std::vector<std::string>& stops) {
        for (const std::string& s : stops) {
            if (!s.empty()) {
                max_len_ = std::max(max_len_, s.size());
                stops_.push_back(s);
            }
        }
    }

    Result push(const std::string& text) {
        Result result;
        if (stops_.empty()) {
            result.emit = text;
            return result;
        }
        buffer_ += text;
        std::size_t best = std::string::npos;
        for (const std::string& s : stops_) {
            const std::size_t pos = buffer_.find(s);
            if (pos != std::string::npos) { best = std::min(best, pos); }
        }
        if (best != std::string::npos) {
            result.emit    = buffer_.substr(0, best);
            result.stopped = true;
            buffer_.clear();
            return result;
        }
        const std::size_t hold = std::min(buffer_.size(), max_len_ - 1);
        // Never cut in the middle of a multibyte UTF-8 sequence: the held bytes
        // complete on the next push (or in flush()).
        const std::size_t emit_len = backup_to_utf8_boundary(buffer_, buffer_.size() - hold);
        result.emit                = buffer_.substr(0, emit_len);
        buffer_.erase(0, emit_len);
        return result;
    }

    std::string flush() {
        std::string out = buffer_;
        buffer_.clear();
        return out;
    }

private:
    std::vector<std::string> stops_;
    std::size_t max_len_ = 1;
    std::string buffer_;
};

} // namespace

ContextOutputBudget resolve_context_output_budget(std::size_t prompt_tokens,
                                                  int requested_max_tokens,
                                                  std::uint32_t max_context) {
    if (requested_max_tokens <= 0) {
        throw std::invalid_argument("requested_max_tokens must be positive");
    }
    if (prompt_tokens > max_context) {
        ApiError error;
        error.status  = 400;
        error.type    = "invalid_request_error";
        error.code    = "context_length_exceeded";
        error.param   = "messages";
        error.message = "prompt (" + std::to_string(prompt_tokens) +
                        " tokens) exceeds max_context (" + std::to_string(max_context) + ")";
        throw ApiException(std::move(error));
    }

    const std::uint64_t available = static_cast<std::uint64_t>(max_context) - prompt_tokens + 1;
    const int effective           = static_cast<int>(
        std::min<std::uint64_t>(static_cast<std::uint64_t>(requested_max_tokens), available));
    return ContextOutputBudget{requested_max_tokens, effective};
}

GenerationService::GenerationService(ServeOptions options) : options_(std::move(options)) {
    caps_.sampling = true; // the engine now honors SamplingParams (temperature 0 == greedy)

    ninfer::EngineOptions engine_options;
    engine_options.device            = options_.device;
    engine_options.max_ctx           = options_.max_context;
    engine_options.prefill_chunk     = options_.prefill_chunk;
    engine_options.mtp_draft_tokens  = options_.mtp_draft_tokens;
    engine_options.kv_dtype          = options_.kv_dtype;
    engine_options.use_cuda_graph    = options_.use_cuda_graph;
    engine_options.use_lm_head_draft = options_.use_lm_head_draft;
    engine_                          = std::make_unique<ninfer::Engine>(engine_options);
    engine_->load(options_.weights_path);
    tokenizer_ = std::make_unique<ninfer::text::QwenTokenizer>(engine_->take_tokenizer_bundle());
    engine_->set_stop_token_ids(tokenizer_->default_stop_token_ids());
}

ninfer::EngineMemoryStats GenerationService::memory_stats() const { return engine_->memory_stats(); }

PreparedRequest GenerationService::prepare(const GenerationRequest& req) const {
    PreparedRequest prepared;
    const std::vector<ninfer::text::ChatMessage> messages = to_chat_messages(req);
    prepared.options                                   = to_generation_options(req, options_);
    prepared.stop_strings                              = req.stop_strings;
    prepared.include_usage                             = req.include_usage;
    prepared.tool_capable                              = req.uses_tools() || req.has_tool_history();

    ninfer::text::ChatRenderOptions render_options = prepared.options.render_options;
    render_options.enable_thinking              = prepared.options.enable_thinking;
    const auto render_start                     = std::chrono::steady_clock::now();
    std::vector<int> ids;
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const ninfer::text::ChatMessage& message) { return message.has_media(); });
    if (has_media) {
        prepared.media_permit = std::unique_lock<std::mutex>(media_mutex_);
        try {
            ninfer::model::Processor processor(*tokenizer_);
            prepared.multimodal.emplace(processor.process(messages, render_options));
        } catch (const ninfer::model::ProcessorError& exception) {
            throw_processor_error(exception);
        } catch (const std::invalid_argument& exception) {
            ApiError error;
            error.status  = 400;
            error.type    = "invalid_request_error";
            error.code    = "invalid_media";
            error.param   = "messages";
            error.message = exception.what();
            throw ApiException(std::move(error));
        }
        ids = prepared.multimodal->input_ids;
    } else {
        const std::string prompt = ninfer::text::render_qwen_chat(messages, render_options);
        ids                      = tokenizer_->encode(prompt);
    }
    prepared.render_tokenize_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - render_start).count();
    prepared.prompt_tokens = static_cast<int>(ids.size());
    // Assistant-content boundary for cross-turn GDN prefix reuse: the position right after the
    // final
    // `<|im_start|>assistant\n` header (before the generation-prompt opener the template appended).
    const std::uint32_t opener =
        ninfer::text::generation_prompt_opener_tokens(*tokenizer_, render_options);
    prepared.content_boundary = opener <= ids.size()
                                    ? static_cast<std::uint32_t>(ids.size()) - opener
                                    : static_cast<std::uint32_t>(ids.size());

    const ContextOutputBudget budget = resolve_context_output_budget(
        ids.size(), prepared.options.max_new_tokens, engine_->max_context());
    prepared.options.max_new_tokens = budget.effective_max_tokens;
    prepared.prompt_token_ids       = std::move(ids);
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& req) const {
    const std::vector<ninfer::text::ChatMessage> messages = to_chat_messages(req);
    ninfer::text::TextGenerationOptions options           = to_generation_options(req, options_);
    ninfer::text::ChatRenderOptions render_options        = options.render_options;
    render_options.enable_thinking                     = options.enable_thinking;
    if (std::any_of(messages.begin(), messages.end(),
                    [](const ninfer::text::ChatMessage& message) { return message.has_media(); })) {
        std::unique_lock<std::mutex> media_permit(media_mutex_);
        try {
            ninfer::model::ProcessorOptions processor_options;
            processor_options.max_prompt_tokens = std::numeric_limits<std::size_t>::max();
            ninfer::model::Processor processor(*tokenizer_, processor_options);
            return static_cast<int>(processor.process(messages, render_options).input_ids.size());
        } catch (const ninfer::model::ProcessorError& exception) {
            throw_processor_error(exception);
        } catch (const std::invalid_argument& exception) {
            ApiError error;
            error.status  = 400;
            error.type    = "invalid_request_error";
            error.code    = "invalid_media";
            error.param   = "messages";
            error.message = exception.what();
            throw ApiException(std::move(error));
        }
    }
    const std::string prompt = ninfer::text::render_qwen_chat(messages, render_options);
    return static_cast<int>(tokenizer_->encode(prompt).size());
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink) {
    std::lock_guard<std::mutex> lock(engine_mutex_);

    ninfer::text::TextGenerationRunner runner(*tokenizer_, *engine_);
    ninfer::text::TextGenerationOptions opt = prepared.options;

    // The runner already splits the <think> block from the answer and tags each
    // delta with its channel; stop strings apply to the answer channel only.
    StopSequences matcher(prepared.stop_strings);
    bool stop_matched = false;
    std::string streamed_answer;

    if (sink != nullptr) {
        opt.stream_callback = [&](const ninfer::text::TextStreamChunk& chunk) {
            if (chunk.channel == ninfer::text::TextChannel::Reasoning) {
                if (sink->on_reasoning) { sink->on_reasoning(chunk.text); }
                return;
            }
            if (stop_matched) { return; }
            const StopSequences::Result r = matcher.push(chunk.text);
            if (!r.emit.empty()) {
                if (prepared.tool_capable) {
                    streamed_answer += r.emit;
                } else if (sink->on_content) {
                    sink->on_content(r.emit);
                }
            }
            if (r.stopped) { stop_matched = true; }
        };
        opt.should_cancel = [&]() {
            return stop_matched || (sink->is_cancelled && sink->is_cancelled());
        };
    }

    // Reset speculative-decoding counters so mtp_stats() reflects only this request;
    // reset + read happen under the same engine lock, so the numbers are per-request.
    engine_->reset_mtp_stats();
    // The prompt was already rendered + tokenized in prepare(); reuse those ids so
    // the chat template renders exactly once per request.
    const ninfer::text::TextGenerationResult result =
        prepared.multimodal ? runner.generate(*prepared.multimodal, opt, prepared.content_boundary,
                                              [&] {
                                                  if (prepared.media_permit.owns_lock()) {
                                                      prepared.media_permit.unlock();
                                                  }
                                              })
                            : runner.generate(std::span<const int>(prepared.prompt_token_ids), opt,
                                              prepared.content_boundary);
    const ninfer::EngineMtpStats mtp = engine_->mtp_stats();

    GenerationOutcome outcome;
    outcome.prompt_tokens                   = static_cast<int>(result.prompt_token_ids.size());
    outcome.completion_tokens               = static_cast<int>(result.generated_token_ids.size());
    outcome.metrics.render_tokenize_seconds = prepared.render_tokenize_seconds;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.total_seconds           = result.timings.total_seconds;
    outcome.metrics.mtp_enabled             = mtp.enabled;
    outcome.metrics.mtp_rounds              = mtp.rounds;
    outcome.metrics.mtp_draft_tokens        = mtp.draft_tokens;
    outcome.metrics.mtp_accepted_tokens     = mtp.accepted_tokens;
    outcome.metrics.prefix_cache_hit_tokens = engine_->last_prefix_cache_hit();

    if (sink != nullptr) {
        if (!stop_matched) {
            const std::string rest = matcher.flush();
            if (!rest.empty()) {
                if (prepared.tool_capable) {
                    streamed_answer += rest;
                } else if (sink->on_content) {
                    sink->on_content(rest);
                }
            }
        }
        outcome.finish_reason = stop_matched ? ninfer::text::FinishReason::Stop : result.finish_reason;
        if (prepared.tool_capable) {
            ParsedToolCallOutput parsed = parse_qwen_tool_call_output(streamed_answer);
            outcome.text                = std::move(parsed.content);
            if (parsed.is_tool_call_response) { outcome.tool_calls = std::move(parsed.tool_calls); }
        }
    } else {
        outcome.reasoning             = result.reasoning;
        const StopSequences::Result r = matcher.push(result.text);
        std::string answer            = r.emit;
        if (!r.stopped) { answer += matcher.flush(); }
        if (prepared.tool_capable) {
            ParsedToolCallOutput parsed = parse_qwen_tool_call_output(answer);
            outcome.text                = std::move(parsed.content);
            if (parsed.is_tool_call_response) { outcome.tool_calls = std::move(parsed.tool_calls); }
        } else {
            outcome.text = std::move(answer);
        }
        outcome.finish_reason = r.stopped ? ninfer::text::FinishReason::Stop : result.finish_reason;
    }
    return outcome;
}

void GenerationService::warmup() {
    try {
        GenerationRequest req;
        req.model = options_.model_id;
        ChatTurn turn;
        turn.role = "user";
        turn.content.push_back(ContentPart{ContentKind::Text, "hi", "text"});
        req.messages.push_back(std::move(turn));
        req.max_tokens           = 4;
        req.max_tokens_set       = true;
        PreparedRequest prepared = prepare(req);
        run(prepared, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "warmup failed (continuing): " << e.what() << '\n';
    }
}

} // namespace ninfer::serve
