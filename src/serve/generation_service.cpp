#include "qus/serve/generation_service.h"

#include "qus/serve/openai_schema.h"
#include "qus/serve/translate.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>

namespace qus::serve {
namespace {

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
        const std::size_t hold     = std::min(buffer_.size(), max_len_ - 1);
        const std::size_t emit_len = buffer_.size() - hold;
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

GenerationService::GenerationService(ServeOptions options) : options_(std::move(options)) {
    tokenizer_ = std::make_unique<qus::text::QwenTokenizer>(options_.tokenizer_path);

    qus::EngineOptions engine_options;
    engine_options.device           = options_.device;
    engine_options.max_ctx          = options_.max_context;
    engine_options.prefill_chunk    = options_.prefill_chunk;
    engine_options.mtp_draft_tokens = options_.mtp_draft_tokens;
    engine_options.use_cuda_graph   = options_.use_cuda_graph;
    engine_options.use_lm_head_draft = options_.use_lm_head_draft;
    engine_options.stop_token_ids   = tokenizer_->default_stop_token_ids();

    engine_ = std::make_unique<qus::Engine>(engine_options);
    engine_->load(options_.weights_path);
}

PreparedRequest GenerationService::prepare(const GenerationRequest& req) const {
    PreparedRequest prepared;
    prepared.messages      = to_chat_messages(req);
    prepared.options       = to_generation_options(req, options_.enable_thinking);
    prepared.stop_strings  = req.stop_strings;
    prepared.include_usage = req.include_usage;

    const std::string prompt = qus::text::render_qwen_chat(
        prepared.messages,
        qus::text::ChatRenderOptions{.enable_thinking = prepared.options.enable_thinking});
    const std::vector<int> ids = tokenizer_->encode(prompt);
    prepared.prompt_tokens     = static_cast<int>(ids.size());

    const std::size_t required =
        ids.size() + static_cast<std::size_t>(std::max(0, prepared.options.max_new_tokens - 1));
    if (required > engine_->max_context()) {
        ApiError error;
        error.status  = 400;
        error.type    = "invalid_request_error";
        error.code    = "context_length_exceeded";
        error.param   = "messages";
        error.message = "prompt (" + std::to_string(ids.size()) +
                        " tokens) plus max_tokens exceeds max_context (" +
                        std::to_string(engine_->max_context()) + ")";
        throw ApiException(std::move(error));
    }
    return prepared;
}

GenerationOutcome GenerationService::run(const PreparedRequest& prepared, const StreamSink* sink) {
    std::lock_guard<std::mutex> lock(engine_mutex_);

    qus::text::TextGenerationRunner runner(*tokenizer_, *engine_);
    qus::text::TextGenerationOptions opt = prepared.options;

    StopSequences matcher(prepared.stop_strings);
    bool stop_matched = false;

    if (sink != nullptr) {
        opt.stream_callback = [&](const qus::text::TextStreamChunk& chunk) {
            const StopSequences::Result r = matcher.push(chunk.text);
            if (!r.emit.empty() && sink->on_delta) { sink->on_delta(r.emit); }
            if (r.stopped) { stop_matched = true; }
        };
        opt.should_cancel = [&]() {
            return stop_matched || (sink->is_cancelled && sink->is_cancelled());
        };
    }

    const qus::text::TextGenerationResult result = runner.generate(prepared.messages, opt);

    GenerationOutcome outcome;
    outcome.prompt_tokens     = static_cast<int>(result.prompt_token_ids.size());
    outcome.completion_tokens = static_cast<int>(result.generated_token_ids.size());

    if (sink != nullptr) {
        if (!stop_matched) {
            const std::string tail = matcher.flush();
            if (!tail.empty() && sink->on_delta) { sink->on_delta(tail); }
        }
        outcome.finish_reason = stop_matched ? qus::text::FinishReason::Stop : result.finish_reason;
    } else {
        const StopSequences::Result r = matcher.push(result.text);
        std::string text              = r.emit;
        if (!r.stopped) { text += matcher.flush(); }
        outcome.text          = std::move(text);
        outcome.finish_reason = r.stopped ? qus::text::FinishReason::Stop : result.finish_reason;
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
        req.max_tokens     = 4;
        req.max_tokens_set = true;
        PreparedRequest prepared = prepare(req);
        run(prepared, nullptr);
    } catch (const std::exception& e) {
        std::cerr << "warmup failed (continuing): " << e.what() << '\n';
    }
}

} // namespace qus::serve
