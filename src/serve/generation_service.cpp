#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/tool_call_parser.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 413;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.code   = "media_fetch_timeout";
        break;
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception,
                                      const char* code = "invalid_media") {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

bool has_media(const GenerationRequest& request) {
    for (const ChatTurn& message : request.messages) {
        for (const ContentPart& part : message.content) {
            if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { return true; }
        }
    }
    return false;
}

ninfer::OwnedMedia acquire_media(const ContentPart& part) {
    ninfer::product::media_acquire::Policy policy;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }

    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes = std::move(source_bytes);
    return media;
}

bool is_processor_budget_error(const std::runtime_error& exception) {
    const std::string_view message(exception.what());
    return message.find("exceeds processor budget") != std::string_view::npos;
}

[[noreturn]] void throw_context_error(const std::invalid_argument& exception) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = "context_length_exceeded";
    error.message = exception.what();
    throw ApiException(std::move(error));
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    ServiceOutputSink(const StreamSink& sink, bool hold_content)
        : sink_(&sink), hold_content_(hold_content) {}

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else if (!hold_content_ && sink_->on_content) {
            sink_->on_content(delta.text);
        }
    }

private:
    const StreamSink* sink_ = nullptr;
    bool hold_content_      = false;
};

} // namespace

GenerationService::GenerationService(ServeOptions options) : options_(std::move(options)) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path            = options_.artifact_path;
    engine_options.device                   = options_.device;
    engine_options.max_context              = options_.max_context;
    engine_options.prefill_chunk            = options_.prefill_chunk;
    engine_options.kv_cache                 = options_.kv_cache;
    engine_options.use_cuda_graph           = options_.use_cuda_graph;
    engine_options.speculative.draft_tokens = static_cast<std::uint32_t>(options_.mtp_draft_tokens);
    engine_options.speculative.proposal_head = options_.proposal_head;
    engine_ = std::make_unique<ninfer::Engine>(std::move(engine_options));
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request) const {
    PreparedRequest prepared;
    prepared.options       = to_request_options(request, options_);
    prepared.include_usage = request.include_usage;
    prepared.tool_capable  = request.uses_tools() || request.has_tool_history();
    if (has_media(request)) { prepared.media_permit = std::unique_lock<std::mutex>(media_mutex_); }

    const auto start = Clock::now();
    try {
        ninfer::PromptInput input = to_prompt_input(
            request, options_, [](const ContentPart& part) { return acquire_media(part); });
        prepared.prompt = engine_->prepare(std::move(input));
    } catch (const ApiException&) { throw; } catch (const std::invalid_argument& exception) {
        if (std::string_view(exception.what()).find("context capacity") != std::string_view::npos) {
            throw_context_error(exception);
        }
        throw_invalid_input(exception);
    } catch (const std::runtime_error& exception) {
        if (is_processor_budget_error(exception)) {
            ApiError error;
            error.status  = 413;
            error.param   = "messages";
            error.code    = "media_budget_exceeded";
            error.message = exception.what();
            throw ApiException(std::move(error));
        }
        throw;
    }
    prepared.prepare_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    prepared.prompt_tokens   = static_cast<int>(prepared.prompt.summary().prompt_tokens);
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request) const {
    std::unique_lock<std::mutex> media_permit;
    if (has_media(request)) { media_permit = std::unique_lock<std::mutex>(media_mutex_); }
    try {
        ninfer::PromptInput input = to_prompt_input(
            request, options_, [](const ContentPart& part) { return acquire_media(part); });
        return static_cast<int>(engine_->count_tokens(std::move(input)));
    } catch (const ApiException&) { throw; } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception);
    } catch (const std::runtime_error& exception) {
        if (is_processor_budget_error(exception)) {
            ApiError error;
            error.status  = 413;
            error.param   = "messages";
            error.code    = "media_budget_exceeded";
            error.message = exception.what();
            throw ApiException(std::move(error));
        }
        throw;
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) {
        output_sink = std::make_unique<ServiceOutputSink>(*sink, prepared.tool_capable);
    }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (sink != nullptr && sink->is_cancelled) {
        cancellation = ninfer::CancellationView(sink->is_cancelled);
    }

    ninfer::GenerationResult result;
    try {
        result = engine_->generate(std::move(prepared.prompt), prepared.options, public_sink,
                                   cancellation);
    } catch (const std::invalid_argument& exception) {
        const std::string_view message(exception.what());
        if (message.find("context capacity") != std::string_view::npos) {
            throw_context_error(exception);
        }
        throw;
    }
    if (prepared.media_permit.owns_lock()) { prepared.media_permit.unlock(); }

    GenerationOutcome outcome;
    outcome.text              = std::move(result.content);
    outcome.reasoning         = std::move(result.reasoning);
    outcome.prompt_tokens     = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens = static_cast<int>(result.generated_token_ids.size());
    outcome.finish_reason     = result.finish_reason;

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.vision_seconds  = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds  = result.timings.decode_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.prefix_cache_hit_tokens = result.reused_prompt_tokens;
    outcome.metrics.mtp_enabled             = result.speculative.enabled;
    outcome.metrics.mtp_rounds              = result.speculative.rounds;
    outcome.metrics.mtp_draft_tokens        = result.speculative.drafted_tokens;
    outcome.metrics.mtp_accepted_tokens     = result.speculative.accepted_tokens;

    if (prepared.tool_capable) {
        ParsedToolCallOutput parsed = parse_qwen_tool_call_output(outcome.text);
        outcome.text                = std::move(parsed.content);
        if (parsed.is_tool_call_response) { outcome.tool_calls = std::move(parsed.tool_calls); }
    }
    return outcome;
}

void GenerationService::warmup() {
    try {
        GenerationRequest request;
        request.model = options_.model_id;
        ChatTurn turn;
        turn.role = "user";
        ContentPart content;
        content.kind     = ContentKind::Text;
        content.text     = "hi";
        content.type_raw = "text";
        turn.content.push_back(std::move(content));
        request.messages.push_back(std::move(turn));
        request.max_tokens       = 4;
        request.max_tokens_set   = true;
        PreparedRequest prepared = prepare(request);
        run(prepared, nullptr);
    } catch (const std::exception& exception) {
        std::cerr << "warmup failed (continuing): " << exception.what() << '\n';
    }
}

} // namespace ninfer::serve
