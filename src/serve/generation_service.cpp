#include "ops/kernel/yarn.h"
#include "ops/launcher/rope.h"
#include "serve/generation_service.h"
#include "serve/prompt_overlap.h"
#include "serve/turn_anchors.h"

#include "product/media_acquire/acquire.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

namespace {
// The grammar vendor prefixes its errors with a timestamp and the server's source path
// ("[21:04:30] /home/.../regex_converter.cc:85: Regex parsing error..."): clients get the message
// without them (capy-flash, second audit N6).
std::string client_message(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    std::size_t at    = 0;
    const auto digits = [&](std::size_t from, std::size_t count) {
        if (from + count > text.size()) return false;
        for (std::size_t i = 0; i < count; ++i)
            if (text[from + i] < '0' || text[from + i] > '9') return false;
        return true;
    };
    while (at < text.size()) {
        // "[hh:mm:ss] " + a path without spaces + ":<line>: "
        if (text[at] == '[' && digits(at + 1, 2) && text.size() > at + 10 && text[at + 3] == ':' &&
            digits(at + 4, 2) && text[at + 6] == ':' && digits(at + 7, 2) &&
            text.substr(at + 9, 2) == "] ") {
            std::size_t end = text.find(' ', at + 11);
            if (end == std::string_view::npos) end = text.size();
            const auto location = text.substr(at + 11, end - at - 11);
            const auto colon = location.rfind(':', location.size() >= 2 ? location.size() - 2 : 0);
            if (colon != std::string_view::npos && location.ends_with(":") &&
                digits(at + 11 + colon + 1, location.size() - colon - 2)) {
                at = std::min(text.size(), end + 1);
                continue;
            }
        }
        result += text[at++];
    }
    return result;
}
} // namespace

ApiError request_error_to_api_error(const ninfer::RequestError& exception,
                                    std::string_view constraint_param) {
    ApiError error;
    error.param   = "messages";
    error.message = client_message(exception.what());
    switch (exception.kind()) {
    // Structured output (Neroued/ninfer 08902f73, eb6696ae, 8ffa3131, 2734a56e). The param is the
    // wire field of the constraint; JSON Schema errors append their JSON Pointer to it.
    case ninfer::RequestErrorKind::UnsupportedOutputCombination:
        error.status = 400;
        error.code   = "unsupported_output_combination";
        error.param =
            exception.pointer().empty() ? std::string(constraint_param) : exception.pointer();
        break;
    case ninfer::RequestErrorKind::InvalidToolOutput:
        error.param.clear();
        error.status = 500;
        error.type   = "server_error";
        error.code   = "invalid_tool_output";
        break;
    case ninfer::RequestErrorKind::InvalidToolConstraint:
        error.status = 400;
        error.code   = "invalid_tool_constraint";
        error.param  = "tools" + exception.pointer();
        break;
    case ninfer::RequestErrorKind::InvalidGrammar:
    case ninfer::RequestErrorKind::InvalidChoice:
    case ninfer::RequestErrorKind::InvalidRegex:
    case ninfer::RequestErrorKind::ConstraintDeadEnd:
        error.status = 400;
        error.param  = constraint_param;
        error.code =
            exception.kind() == ninfer::RequestErrorKind::InvalidGrammar  ? "invalid_grammar"
            : exception.kind() == ninfer::RequestErrorKind::InvalidChoice ? "invalid_choice"
            : exception.kind() == ninfer::RequestErrorKind::InvalidRegex  ? "invalid_regex"
                                                                          : "constraint_dead_end";
        break;
    // capy-flash limits (OutputConstraintLimits): an input or compilation over a limit, or a
    // generating request whose matcher exceeded its time budget (only that request fails).
    case ninfer::RequestErrorKind::ConstraintLimitExceeded:
        error.status = 400;
        error.code   = "constraint_limit_exceeded";
        error.param  = std::string(constraint_param) + exception.pointer();
        break;
    case ninfer::RequestErrorKind::ConstraintBudgetExceeded:
        error.status = 400;
        error.code   = "constraint_budget_exceeded";
        error.param  = constraint_param;
        break;
    case ninfer::RequestErrorKind::InvalidJsonSchema:
    case ninfer::RequestErrorKind::UnsupportedJsonSchema:
    case ninfer::RequestErrorKind::UnsatisfiableJsonSchema:
        error.status = 400;
        error.param  = exception.source() == ninfer::RequestErrorSource::Tools
                           ? "tools" + exception.pointer()
                           : std::string(constraint_param) + exception.pointer();
        error.code   = exception.kind() == ninfer::RequestErrorKind::InvalidJsonSchema
                           ? "invalid_json_schema"
                       : exception.kind() == ninfer::RequestErrorKind::UnsupportedJsonSchema
                           ? "unsupported_json_schema"
                           : "unsatisfiable_json_schema";
        break;
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
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
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception,
                                      std::string_view constraint_param = {}) {
    throw ApiException(request_error_to_api_error(exception, constraint_param));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart start) override {
        if (sink_->on_start) { sink_->on_start(start); }
    }

    void progress(ninfer::PromptProgress progress) override {
        if (sink_->on_progress) { sink_->on_progress(progress); }
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        if (sink_->on_timing) { sink_->on_timing(timing); }
    }

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

// The continuation of a reasoning-close retry publishes into the stream the client already
// receives: its own start, progress and timing facts are not forwarded (the first reply's are
// the response's), only its output deltas.
class ContinuationOutputSink final : public ninfer::OutputSink {
public:
    explicit ContinuationOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart) override {}

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer)
    : options_(std::move(options)) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path          = options_.artifact_path;
    engine_options.chat_template_path     = options_.chat_template_path;
    engine_options.device                 = options_.device;
    engine_options.tp                     = options_.tp;
    engine_options.devices                = options_.devices;
    engine_options.max_context            = options_.max_context;
    engine_options.kv_capacity            = options_.kv_capacity;
    engine_options.max_concurrency        = options_.max_concurrency;
    engine_options.max_pending_requests   = options_.max_pending_requests;
    engine_options.pending_timeout_ms     = options_.pending_timeout_ms;
    engine_options.decode_share           = options_.decode_share;
    engine_options.prefill_chunk          = options_.prefill_chunk;
    engine_options.kv_lease_window_tokens = options_.kv_lease_window_tokens;
    engine_options.kv_cache               = options_.kv_cache;
    engine_options.elastic_kv             = options_.elastic_kv;
    engine_options.kv_ram_active          = options_.kv_ram_active;
    if (options_.miss_split) {
        engine_options.routed_experts.miss_split.mode      = *options_.miss_split;
        engine_options.routed_experts.miss_split.calibrate = options_.miss_split_calibrate;
    }
    if (options_.expert_cpu_threads) {
        engine_options.routed_experts.cpu_threads      = *options_.expert_cpu_threads;
        engine_options.routed_experts.gguf_cpu_threads = *options_.expert_cpu_threads;
    }
    if (options_.stream_min_tokens) {
        engine_options.routed_experts.stream_min_tokens = *options_.stream_min_tokens;
    }
    engine_options.rope_yarn_factor         = options_.rope_yarn_factor;
    engine_options.enable_vision            = options_.enable_vision;
    engine_options.vision_device            = options_.vision_device;
    engine_options.max_vision_tokens        = options_.max_vision_tokens;
    engine_options.use_cuda_graph           = options_.use_cuda_graph;
    engine_options.tp_mailbox               = options_.tp_mailbox;
    engine_options.embedding_host           = options_.embedding_host;
    engine_options.host_pinned_max_bytes    = options_.host_pinned_max_bytes;
    engine_options.speculative              = options_.speculative;
    engine_options.context_cache            = options_.context_cache;
    engine_options.context_cost.preset_path = options_.context_cost_presets;
    engine_options.media_cache_bytes        = options_.media_cache_bytes;
    engine_options.media_live_bytes         = options_.media_live_bytes;
    engine_options.media_preprocess_threads = options_.media_preprocess_threads;
    engine_options.constraint_limits        = options_.constraint_limits;
    engine_options.startup_observer         = std::move(startup_observer);
    engine_ = std::make_unique<ninfer::Engine>(std::move(engine_options));
    // YaRN tables go in BEFORE the first forward pass: the rope kernel reads the __constant__
    // table at launch, so installing after construction is correct and factor 1 never calls this.
    if (options_.rope_yarn_factor > 1.0F) {
        ninfer::rope::YarnParams yarn;
        yarn.factor                = options_.rope_yarn_factor;
        yarn.original_max_position = static_cast<int>(options_.rope_original_max_position);
        // rotary_dim 64 and rope_theta 1e7 are this family's geometry: the 27B and Flash-Next both
        // declare partial_rotary_factor 0.25 on head_dim 256, i.e. 32 frequency pairs.
        const auto freqs = ninfer::rope::yarn_inv_freq(yarn, yarn.rotary_dim);
        ninfer::ops::detail::rope_install_text_inv_frequency(freqs.data(),
                                                             static_cast<int>(freqs.size()));
        ninfer::ops::detail::rope_install_text_attention_scale(
            ninfer::rope::yarn_attention_scale(yarn));
    }
    turn_anchors_ =
        resolve_turn_anchors(options_.turn_anchors, engine_->load_summary().architecture);
    if (options_.reasoning_close_retry) {
        reasoning_guard_tokens_ = engine_->reasoning_guard_tokens();
    }
    // On by default since v0.7: a model whose stop configuration the guard cannot handle (no or
    // more than two terminal ids, no thinking opener) serves without it instead of not starting.
    try {
        engine_->configure_reasoning_eos_guard(options_.reasoning_eos_guard);
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "reasoning EOS guard off for this model: %s\n", error.what());
        engine_->configure_reasoning_eos_guard(false);
    }
    request_capacity_ = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    if (!options_.prefill_miss_dump_dir.empty()) {
        const ninfer::Engine* engine = engine_.get();
        prefill_miss_dump_           = std::make_shared<PrefillMissDump>(
            PrefillMissDumpOptions{.directory  = options_.prefill_miss_dump_dir,
                                             .min_tokens = options_.prefill_miss_min_tokens,
                                             .min_match  = options_.prefill_miss_min_match},
            [engine](std::span<const ninfer::TokenId> tokens) {
                return engine->detokenize_text(tokens);
            });
    }
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(request_capacity_, started, deadline);
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninfer::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    if (options_.log_speculation_detail) {
        observation.accepted_length_steps = options_.speculation_detail_max_steps;
    }
    return prepare_impl(
        request, consumer_mode, observation, std::move(is_cancelled), std::move(context_cache),
        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite : CacheParticipation::Disabled,
        DeadlinePolicy::ClientPendingTimeout, options_.log_speculation_detail);
}

PreparedRequest GenerationService::prepare_impl(
    const GenerationRequest& request, GenerationConsumerMode consumer_mode,
    ninfer::GenerationObservationOptions observation, std::function<bool()> is_cancelled,
    ContextCacheHints context_cache, CacheParticipation cache_participation,
    DeadlinePolicy deadline_policy, bool speculation_detail) const {
    PreparedRequest prepared;
    prepared.speculation_detail             = speculation_detail;
    prepared.constraint_param               = request.constraint_param;
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    ninfer::RequestOptions request_options  = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    prepared.thinking_budget     = request_options.execution.thinking.budget;
    prepared.reasoning_effort    = semantics.reasoning_effort;
    prepared.preserve_thinking   = semantics.preserve_thinking;
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy);

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        std::vector<PromptCacheMarker> protocol_markers = std::move(input.context_cache.markers);
        const bool protocol_allows_engine_automatic =
            input.context_cache.allow_engine_automatic_shared_prefixes;
        input.context_cache = std::move(context_cache);
        input.context_cache.markers.insert(input.context_cache.markers.end(),
                                           std::make_move_iterator(protocol_markers.begin()),
                                           std::make_move_iterator(protocol_markers.end()));
        input.context_cache.allow_engine_automatic_shared_prefixes =
            input.context_cache.allow_engine_automatic_shared_prefixes &&
            protocol_allows_engine_automatic;
        add_turn_anchor_markers(input, turn_anchors_);
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        std::shared_ptr<ReasoningRetryContext> retry;
        // A constrained request is never retried: the continuation would run without the
        // grammar state of the first reply.
        if (options_.reasoning_close_retry && !request_has_media && !request.constraint &&
            cache_participation == CacheParticipation::ReadWrite) {
            retry                = std::make_shared<ReasoningRetryContext>();
            retry->input         = input;
            retry->consumer_mode = consumer_mode;
            retry->observation   = observation;
        }
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.enable_thinking = prompt.summary().starts_in_reasoning;
        if (retry && prepared.enable_thinking) {
            retry->request_options   = request_options;
            prepared.reasoning_retry = std::move(retry);
        }
        if (!prepared.enable_thinking) {
            request_options.execution.thinking.budget.reset();
            prepared.thinking_budget.reset();
        }
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        if (speculation_detail || (prefill_miss_dump_ != nullptr &&
                                   cache_participation == CacheParticipation::ReadWrite)) {
            const std::span<const ninfer::TokenId> prompt_ids = prompt.token_ids();
            prepared.prompt_token_ids.assign(prompt_ids.begin(), prompt_ids.end());
        }
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? ninfer::OutputConsumerMode::Streaming
                                                  : ninfer::OutputConsumerMode::Aggregate,
                                              observation, prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception, request.constraint_param);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(request, options_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = ninfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }

    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (...) {
        // A request that ends without a result still leaves its prompt as a context to compare
        // the next one against.
        if (prefill_miss_dump_ != nullptr && !prepared.prompt_token_ids.empty()) {
            (void)prefill_miss_dump_->observe(
                prepared.prompt_token_ids, {},
                static_cast<std::uint32_t>(prepared.prompt_token_ids.size()), "failed");
        }
        try {
            throw;
        } catch (const ninfer::RequestError& exception) {
            throw_request_error(exception, prepared.constraint_param);
        }
    }
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds          = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    outcome.metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.engine_timing               = result.engine_timing;
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.materialization             = result.materialization;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);
    if (prepared.speculation_detail) {
        outcome.metrics.speculative_accepted_lengths =
            std::move(result.speculative.accepted_lengths);
        outcome.metrics.prompt_ngram_overlap =
            prompt_ngram_overlap(prepared.prompt_token_ids, result.generated_token_ids);
    }
    if (prefill_miss_dump_ != nullptr && !prepared.prompt_token_ids.empty()) {
        outcome.metrics.prefill_miss = prefill_miss_dump_->observe(
            prepared.prompt_token_ids, result.generated_token_ids, result.reused_prompt_tokens,
            result.finish_reason == ninfer::FinishReason::Cancelled ? "cancelled" : "finished");
    }
    prepared.prompt_token_ids = {};

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
    if (prepared.reasoning_retry) {
        apply_reasoning_retry(prepared, result, outcome, sink, cancellation);
        prepared.reasoning_retry.reset();
    }
    if (options_.reasoning_eos_guard) {
        outcome.reasoning_eos_guard_total = engine_->reasoning_eos_guard_replacements();
    }
    return outcome;
}

void GenerationService::apply_reasoning_retry(PreparedRequest& prepared,
                                              const ninfer::GenerationResult& first,
                                              GenerationOutcome& outcome, const StreamSink* sink,
                                              const ninfer::CancellationView& cancellation) {
    ReasoningRetryContext& retry  = *prepared.reasoning_retry;
    const ReasoningRetryPlan plan = plan_reasoning_retry(
        ReasoningRetryInput{.starts_in_reasoning      = prepared.enable_thinking,
                            .finish_reason            = outcome.finish_reason,
                            .generated                = first.generated_token_ids,
                            .content                  = outcome.text,
                            .tool_calls               = outcome.tool_calls.size(),
                            .thinking_control_applied = outcome.thinking.applied,
                            .requested_output_tokens =
                                retry.request_options.execution.requested_output_tokens},
        reasoning_guard_tokens_);
    if (plan.trigger == ReasoningRetryTrigger::None) { return; }
    if (cancellation.requested()) { return; }
    ReasoningRetryStats& stats = outcome.reasoning_retry;
    stats.trigger              = plan.trigger;
    stats.kept_tokens          = plan.kept_tokens;
    stats.injected_tokens      = plan.injected_tokens;

    ninfer::RequestOptions options = std::move(retry.request_options);
    if (plan.remaining_output_tokens) {
        options.execution.requested_output_tokens = *plan.remaining_output_tokens;
    }
    // A fresh draw: the continuation of an EmptyAfterClose reply samples the same position again.
    options.execution.sampling.seed = prepared.sampling.seed + 1U;
    options.execution.thinking.budget.reset();
    std::unique_ptr<ContinuationOutputSink> continuation_sink;
    if (sink != nullptr) { continuation_sink = std::make_unique<ContinuationOutputSink>(*sink); }
    ninfer::GenerationResult second;
    try {
        const auto deadline = Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
        const PreparationControl control{.deadline = deadline, .cancellation = cancellation};
        ninfer::PreparedPrompt prompt =
            engine_->prepare_continuation(std::move(retry.input), plan.appended, control);
        ninfer::GenerationHandle generation =
            engine_->submit(std::move(prompt), std::move(options),
                            retry.consumer_mode == GenerationConsumerMode::Streaming
                                ? ninfer::OutputConsumerMode::Streaming
                                : ninfer::OutputConsumerMode::Aggregate,
                            retry.observation, deadline);
        second = generation.wait(continuation_sink.get(), cancellation);
    } catch (const std::exception& exception) {
        // The first reply stands as it was.
        stats.error = exception.what();
        return;
    }

    stats.continuation_prompt     = second.prompt.prompt_tokens;
    stats.continuation_reused     = second.reused_prompt_tokens;
    stats.continuation_completion = static_cast<std::uint32_t>(second.generated_token_ids.size());
    stats.continuation_path       = second.prefix_reuse_path;

    outcome.text = std::move(second.content);
    outcome.reasoning += second.reasoning;
    outcome.tool_calls          = std::move(second.tool_calls);
    outcome.tool_call_parse     = second.tool_call_parse;
    outcome.finish_reason       = second.finish_reason;
    outcome.matched_stop_string = std::move(second.matched_stop_string);
    outcome.completion_tokens   = static_cast<int>(plan.kept_tokens + plan.injected_tokens +
                                                   second.generated_token_ids.size());
    outcome.reasoning_tokens += static_cast<int>(second.reasoning_tokens);

    GenerationMetrics& metrics = outcome.metrics;
    metrics.prefill_seconds += second.timings.prefill_seconds;
    metrics.decode_seconds += second.timings.decode_seconds;
    metrics.prompt_wall_seconds += second.timings.prompt_wall_seconds;
    metrics.generation_wall_seconds += second.timings.generation_wall_seconds;
    metrics.total_seconds += second.timings.total_seconds;
    metrics.speculative_rounds += second.speculative.rounds;
    metrics.speculative_draft_tokens += second.speculative.drafted_tokens;
    metrics.speculative_accepted_tokens += second.speculative.accepted_tokens;
    metrics.speculative_fallback_steps += second.speculative.fallback_steps;
    const auto& positions = second.speculative.accepted_per_position;
    if (metrics.speculative_accepted_per_position.size() < positions.size()) {
        metrics.speculative_accepted_per_position.resize(positions.size(), 0);
    }
    for (std::size_t index = 0; index < positions.size(); ++index) {
        metrics.speculative_accepted_per_position[index] += positions[index];
    }
}

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    PreparedRequest prepared =
        prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                     CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup, false);
    run(prepared, nullptr);
}

} // namespace ninfer::serve
