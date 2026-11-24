#include "ninfer/engine.h"
#include "context_budget.h"
#include "kv_cache_storage.h"
#include "symbol_interpose.h"

#include <cuda_runtime_api.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

std::atomic<bool> fail_next_event_query{false};
std::atomic<bool> event_failure_injected{false};

} // namespace

extern "C" cudaError_t CUDARTAPI cudaEventQuery(cudaEvent_t event) {
    static const auto real =
        ninfer::test::next_symbol<cudaError_t (*)(cudaEvent_t)>("cudaEventQuery");
    if (fail_next_event_query.exchange(false)) {
        event_failure_injected.store(true);
        return cudaErrorInvalidResourceHandle;
    }
    return real(event);
}

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.prefill_chunk             = 1024;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = true;
    options.max_concurrency           = 1;
    options.max_pending_requests      = 1;
    ninfer::test::qwen3_5::set_context_capacity(options, 4);
    return options;
}

ninfer::EngineOptions host_restore_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 512;
    options.prefill_chunk             = 256;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.max_concurrency           = 1;
    options.max_pending_requests      = 1;
    const auto plan                   = ninfer::test::qwen3_5::context_capacity_plan(options);
    options.context_cache.host_capacity_bytes =
        ninfer::test::qwen3_5::checkpoint_capacity_bytes(plan, 1, options.max_context);
    options.context_capacity = ninfer::ContextCapacityPolicy::explicit_capacity(
        ninfer::test::qwen3_5::context_capacity_bytes(plan));
    return options;
}

ninfer::EngineOptions anthropic_prefix_regression_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 2048;
    options.prefill_chunk             = 512;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.max_concurrency           = 1;
    options.max_pending_requests      = 1;

    // Retain four full checkpoints in Host while Device stays at its startup minimum.
    const auto plan = ninfer::test::qwen3_5::context_capacity_plan(options);
    options.context_cache.host_capacity_bytes =
        ninfer::test::qwen3_5::checkpoint_capacity_bytes(plan, 4, options.max_context);
    options.context_capacity = ninfer::ContextCapacityPolicy::explicit_capacity(
        ninfer::test::qwen3_5::context_capacity_bytes(plan));
    return options;
}

ninfer::EngineOptions shared_rewrite_materialization_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 100000;
    options.prefill_chunk                     = 1024;
    options.kv_cache                          = ninfer::KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
    options.speculative.backend               = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens          = 3;
    options.speculative.proposal_head         = ninfer::ProposalHead::Optimized;
    options.max_concurrency                   = 1;
    options.max_pending_requests              = 1;
    options.context_cache.host_capacity_bytes = 0;
    ninfer::test::qwen3_5::set_context_capacity(options, 2);
    return options;
}

ninfer::EngineOptions explicit_anchor_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 512;
    options.prefill_chunk                     = 256;
    options.speculative.backend               = ninfer::SpeculativeBackend::None;
    options.max_concurrency                   = 1;
    options.max_pending_requests              = 1;
    options.context_cache.host_capacity_bytes = 0;
    ninfer::test::qwen3_5::set_context_capacity(options, 4);
    return options;
}

ninfer::EngineOptions concurrent_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                     = artifact;
    options.max_context                       = 512;
    options.prefill_chunk                     = 256;
    options.speculative.backend               = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens          = 3;
    options.speculative.proposal_head         = ninfer::ProposalHead::Optimized;
    options.max_concurrency                   = 8;
    options.max_pending_requests              = 8;
    options.context_cache.host_capacity_bytes = 0;
    ninfer::test::qwen3_5::set_context_capacity(options, 16);
    return options;
}

std::vector<std::uint8_t> gradient_ppm(int width = 64, int height = 64) {
    std::vector<std::uint8_t> ppm;
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < width * height; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput chinese_chat(bool enable_thinking) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "你好，简单介绍一下你自己。", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = enable_thinking;
    return input;
}

int exercise_artifact_frontend(const ninfer::Engine& engine) {
    if (engine.count_tokens(chinese_chat(true)) != 16) {
        std::cerr << "artifact tokenizer/chat template changed the thinking prompt golden\n";
        return 1;
    }
    if (engine.count_tokens(chinese_chat(false)) != 18) {
        std::cerr << "artifact tokenizer/chat template changed the no-thinking prompt golden\n";
        return 1;
    }
    return 0;
}

class ObservationSink final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart start) override {
        if (started_) { valid_ = false; }
        started_ = true;
        start_   = start;
    }

    void progress(ninfer::PromptProgress progress) override {
        if (!started_ || timing_seen_ ||
            progress.total_prompt_tokens != start_.prompt.prompt_tokens ||
            progress.reused_prompt_tokens != start_.reused_prompt_tokens ||
            progress.processed_prompt_tokens < last_processed_ ||
            progress.processed_prompt_tokens > progress.total_prompt_tokens ||
            progress.elapsed_ns < last_progress_elapsed_ns_) {
            valid_ = false;
        }
        last_processed_           = progress.processed_prompt_tokens;
        last_progress_elapsed_ns_ = progress.elapsed_ns;
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        if (!started_ || last_processed_ != start_.prompt.prompt_tokens ||
            (timing_seen_ && (timing.generated_tokens < last_timing_.generated_tokens ||
                              timing.prompt_elapsed_ns != last_timing_.prompt_elapsed_ns ||
                              timing.generation_elapsed_ns < last_timing_.generation_elapsed_ns))) {
            valid_ = false;
        }
        timing_seen_ = true;
        last_timing_ = timing;
    }

    void publish(ninfer::OutputEvent event) override {
        const auto* text = std::get_if<ninfer::OutputDelta>(&event);
        if (!text) return;
        const auto& delta = *text;
        if (!timing_seen_) { valid_ = false; }
        published_output_ = published_output_ || !delta.text.empty();
    }

    [[nodiscard]] bool valid_for(const ninfer::GenerationResult& result) const {
        return valid_ && started_ && timing_seen_ && start_.reused_prompt_tokens == 0 &&
               last_processed_ == start_.prompt.prompt_tokens &&
               last_timing_.generated_tokens == result.generated_token_ids.size() &&
               result.stats.timings.prompt_wall_seconds > 0.0 &&
               result.stats.timings.generation_wall_seconds >= 0.0;
    }

    [[nodiscard]] bool published_output() const noexcept { return published_output_; }

private:
    ninfer::GenerationStart start_;
    ninfer::GenerationTimingObservation last_timing_;
    std::uint32_t last_processed_           = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    bool started_                           = false;
    bool timing_seen_                       = false;
    bool valid_                             = true;
    bool published_output_                  = false;
};

int exercise_stream_observations(ninfer::Engine& engine) {
    std::vector<ninfer::TokenId> prompt(2050, 198);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 3;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const ninfer::GenerationObservationOptions observation{
        .constraint_timings = true, .live_timings = true, .prompt_progress = true};

    ObservationSink sink;
    ninfer::GenerationHandle generation =
        engine.submit(engine.prepare_tokens(std::move(prompt)), std::move(request),
                      ninfer::OutputConsumerMode::Streaming, observation);
    generation.wait_for_admission();
    generation.wait_for_admission();
    const ninfer::GenerationResult result = generation.wait(&sink);
    if (result.generated_token_ids.size() != 3 || !sink.valid_for(result)) {
        std::cerr
            << "stream observations lost prompt progress, commit timing, or publication order\n";
        return 1;
    }
    if (!sink.published_output() || !result.stats.first_output_timing ||
        result.stats.first_output_timing->elapsed_seconds >
            result.stats.timings.total_seconds - result.stats.timings.prepare_seconds + 1.0e-9 ||
        result.stats.first_output_timing->computed_prefill_tokens !=
            result.stats.computed_prefill_tokens ||
        !result.stats.first_output_timing->pending_execution ||
        !result.stats.first_output_timing->pending_execution->latency.result_ns) {
        std::cerr << "first nonempty stream output lost its request-owned timing boundary\n";
        return 1;
    }

    ninfer::RequestOptions cancellation_request;
    cancellation_request.execution.requested_output_tokens = 3;
    cancellation_request.execution.allow_prefix_reuse      = false;
    ObservationSink cancelled_sink;
    auto cancelled =
        engine.submit(engine.prepare_tokens(std::vector<ninfer::TokenId>(2050, 198)),
                      cancellation_request, ninfer::OutputConsumerMode::Streaming, observation);
    const auto cancelled_result =
        cancelled.wait(&cancelled_sink, ninfer::CancellationView([] { return true; }));
    if (cancelled_result.finish_reason != ninfer::FinishReason::Cancelled ||
        cancelled_sink.published_output() || !cancelled_result.generated_token_ids.empty() ||
        cancelled_result.stats.first_output_timing) {
        std::cerr << "cancellation before output fabricated a first-output snapshot\n";
        return 1;
    }
    return 0;
}

int exercise_admission_wait(ninfer::Engine& engine) {
    const auto prompt = [&] {
        return engine.prepare_tokens({248045, 846, 198, 5834, 248046, 198});
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 4090;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;

    auto active = engine.submit(prompt(), request);
    active.wait_for_admission();
    request.execution.requested_output_tokens = 1;

    auto prepared = prompt();
    auto pending =
        engine.submit(std::move(prepared), request, ninfer::OutputConsumerMode::Streaming, {},
                      std::chrono::steady_clock::now() + std::chrono::milliseconds(30));
    bool timed_out = false;
    try {
        pending.wait_for_admission();
    } catch (const ninfer::RequestError& error) {
        timed_out = error.kind() == ninfer::RequestErrorKind::QueueTimeout;
    }
    ObservationSink pending_sink;
    bool retained_error = false;
    try {
        (void)pending.wait(&pending_sink);
    } catch (const ninfer::RequestError& error) {
        retained_error = error.kind() == ninfer::RequestErrorKind::QueueTimeout;
    }
    if (!timed_out || !retained_error || pending_sink.published_output()) {
        std::cerr << "admission wait lost the pending deadline or consumed the failure\n";
        return 1;
    }
    const auto after_timeout = engine.snapshot().runtime;
    if (after_timeout.waiting_requests != 0 ||
        after_timeout.running_requests + after_timeout.materializing_requests != 1) {
        std::cerr << "pending timeout was visible before its scheduling snapshot\n";
        return 1;
    }

    auto cancelled             = engine.submit(prompt(), request);
    bool cancellation_reported = false;
    try {
        cancelled.wait_for_admission(ninfer::CancellationView([] { return true; }));
    } catch (const ninfer::RequestError& error) {
        cancellation_reported = error.kind() == ninfer::RequestErrorKind::Cancelled;
    }
    const auto cancelled_result = cancelled.wait();
    if (!cancellation_reported ||
        cancelled_result.finish_reason != ninfer::FinishReason::Cancelled ||
        !cancelled_result.generated_token_ids.empty()) {
        std::cerr << "admission cancellation did not settle the pending request\n";
        return 1;
    }
    const auto after_pending_cancel = engine.snapshot().runtime;
    if (after_pending_cancel.waiting_requests != 0 ||
        after_pending_cancel.running_requests + after_pending_cancel.materializing_requests != 1) {
        std::cerr << "pending cancellation changed the surviving request's membership\n";
        return 1;
    }

    request.execution.requested_output_tokens = 0;
    auto immediate                            = engine.submit(prompt(), request);
    immediate.wait_for_admission();
    const auto immediate_result = immediate.wait();
    if (immediate_result.finish_reason != ninfer::FinishReason::OutputLimit ||
        !immediate_result.generated_token_ids.empty()) {
        std::cerr << "admission wait blocked or consumed an immediate result\n";
        return 1;
    }

    const auto stopped = active.wait(nullptr, ninfer::CancellationView([] { return true; }));
    if (stopped.finish_reason != ninfer::FinishReason::Cancelled) {
        std::cerr << "admitted request did not retain its cancellation lifetime\n";
        return 1;
    }
    return 0;
}

void require_no_requests(const ninfer::RuntimeStats& stats) {
    if (stats.running_requests || stats.waiting_requests || stats.materializing_requests ||
        stats.paused_requests || stats.replaying_requests || stats.prefilling_requests ||
        stats.decode_ready_requests || stats.capture_pending_requests ||
        stats.terminal_pending_requests) {
        throw std::runtime_error("terminal response preceded logical membership release");
    }
}

int exercise_terminal_publication(ninfer::Engine& engine) {
    const auto prompt = [&] {
        return engine.prepare_tokens({248045, 846, 198, 5834, 248046, 198});
    };
    ninfer::RequestOptions request;
    request.execution.sampling.temperature = 0.0F;
    request.execution.allow_prefix_reuse   = false;
    request.stop.include_model_defaults    = false;

    for (const bool cancel : {false, true}) {
        const auto before                         = engine.snapshot();
        request.execution.requested_output_tokens = cancel ? 128 : 1;
        bool first_token                          = false;
        ninfer::GenerationObservationOptions observation;
        observation.first_token = [&](const auto&) { first_token = true; };
        auto handle = engine.submit(prompt(), request, ninfer::OutputConsumerMode::Aggregate,
                                    std::move(observation));
        const auto result =
            handle.wait(nullptr, ninfer::CancellationView([&] { return cancel && first_token; }));
        const auto after = engine.snapshot();
        require_no_requests(after.runtime);
        if (!first_token ||
            result.finish_reason !=
                (cancel ? ninfer::FinishReason::Cancelled : ninfer::FinishReason::OutputLimit) ||
            after.stamp.sequence <= before.stamp.sequence ||
            after.runtime.generated_tokens - before.runtime.generated_tokens !=
                result.generated_token_ids.size()) {
            throw std::runtime_error("terminal response lost committed output or its snapshot");
        }
    }
    if (const int result = exercise_admission_wait(engine); result != 0) { return result; }
    require_no_requests(engine.snapshot().runtime);

    const auto features = engine.snapshot().ranks.at(0).memory.vision->max_window_features;
    int side            = 32;
    while (std::uint64_t(side / 32) * (side / 32) <= features) { side += 32; }
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm(side, side);
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "above-window.ppm";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    message.parts.push_back(ninfer::MessagePart{.text = "Describe this image."});
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    input.messages.push_back(std::move(message));
    request.execution.requested_output_tokens = 1;
    auto rejected     = engine.submit(engine.prepare(std::move(input)), request);
    bool budget_error = false;
    try {
        (void)rejected.wait();
    } catch (const ninfer::RequestError& error) {
        budget_error = error.kind() == ninfer::RequestErrorKind::MediaBudgetExceeded;
    }
    require_no_requests(engine.snapshot().runtime);
    if (!budget_error || !engine.is_available()) {
        throw std::runtime_error("native request rejection did not settle independently");
    }

    // Both an admitted and a queued request must observe the same instance failure only after
    // all logical membership has been released. The injected error leaves CUDA usable for cleanup.
    request.execution.requested_output_tokens = 4090;
    ninfer::GenerationObservationOptions failure_observation;
    failure_observation.first_token = [](const auto&) { fail_next_event_query.store(true); };
    auto active = engine.submit(prompt(), request, ninfer::OutputConsumerMode::Aggregate,
                                std::move(failure_observation));
    active.wait_for_admission();
    auto pending = engine.submit(prompt(), request);
    std::string first_error;
    for (auto* handle : {&active, &pending}) {
        std::string error;
        try {
            (void)handle->wait();
        } catch (const std::exception& failure) { error = failure.what(); }
        require_no_requests(engine.snapshot().runtime);
        if (error.find("cudaEventQuery failed") == std::string::npos || engine.is_available()) {
            throw std::runtime_error("instance failure did not preserve its execution error");
        }
        if (first_error.empty()) first_error = error;
        if (error != first_error) {
            throw std::runtime_error("instance failure changed across affected requests");
        }
    }
    if (!event_failure_injected.load()) {
        throw std::runtime_error("terminal failure fixture did not reach execution");
    }
    std::cout << "terminal publication: completion, cancellation, queue/native errors and instance "
                 "failure\n";
    return 0;
}

int exercise_full_prefill_chunk(ninfer::Engine& engine) {
    constexpr std::size_t kChunkTokens = 1024;
    std::vector<ninfer::TokenId> prompt(kChunkTokens, 198);
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(std::move(prompt)), options);
    if (result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "full-chunk prefill did not complete through the planned workspace\n";
        return 1;
    }
    return 0;
}

int exercise_abandoned_handle_capacity(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;

    {
        auto abandoned = engine.submit(engine.prepare_tokens(prompt), request);
        if (!abandoned) {
            std::cerr << "abandonment fixture did not create a generation handle\n";
            return 1;
        }
        abandoned.wait_for_admission();
    }
    const auto crossed = engine.generate(engine.prepare_tokens(prompt), request);
    if (crossed.generated_token_ids.size() != 1) {
        std::cerr << "request after an abandoned handle did not complete\n";
        return 1;
    }

    auto first      = engine.submit(engine.prepare_tokens(prompt), request);
    auto second     = engine.submit(engine.prepare_tokens(prompt), request);
    bool overloaded = false;
    try {
        auto third = engine.submit(engine.prepare_tokens(prompt), request);
        (void)third;
    } catch (const ninfer::RequestError& error) {
        overloaded = error.kind() == ninfer::RequestErrorKind::Overloaded;
    }
    if (!overloaded) {
        std::cerr << "outstanding capacity was released twice or not enforced\n";
        return 1;
    }
    if (first.wait().generated_token_ids.size() != 1 ||
        second.wait().generated_token_ids.size() != 1) {
        std::cerr << "requests retained after the overload check did not complete\n";
        return 1;
    }
    return 0;
}

int exercise_zero_suffix_reuse(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    ninfer::RequestOptions baseline_options;
    baseline_options.execution.requested_output_tokens = 8;
    baseline_options.execution.sampling.temperature    = 0.0F;
    baseline_options.execution.allow_prefix_reuse      = true;
    baseline_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(prompt), baseline_options);
    if (baseline.generated_token_ids.size() != 8) {
        std::cerr << "zero-suffix baseline did not generate eight tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> exact_frontier = prompt;
    exact_frontier.insert(exact_frontier.end(), baseline.generated_token_ids.begin(),
                          baseline.generated_token_ids.end() - 1);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 2;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(exact_frontier), reuse_options);
    if (reused.stats.reused_prompt_tokens != exact_frontier.size()) {
        std::cerr << "zero-suffix reuse count is " << reused.stats.reused_prompt_tokens
                  << ", expected " << exact_frontier.size() << '\n';
        return 1;
    }
    if (reused.generated_token_ids.size() != 2 ||
        reused.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "zero-suffix reuse did not resume from the retained target frontier\n";
        return 1;
    }
    return 0;
}

int exercise_prefix(ninfer::Engine& engine) {
    ninfer::RequestOptions first_options;
    first_options.execution.requested_output_tokens = 5;
    first_options.execution.sampling.temperature    = 0.0F;
    first_options.stop.include_model_defaults       = false;

    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), first_options);
    if (first.generated_token_ids.size() != 5) {
        std::cerr << "first request did not generate five tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 5;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), reuse_options);

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.stats.reused_prompt_tokens != expected_reuse) {
        std::cerr << "append reuse count is " << reused.stats.reused_prompt_tokens << ", expected "
                  << expected_reuse << '\n';
        return 1;
    }

    if (const int result = exercise_zero_suffix_reuse(engine, prompt); result != 0) {
        return result;
    }

    return 0;
}

int exercise_semantic_captures(const char* artifact) {
    auto configured          = host_restore_engine_options(artifact);
    configured.max_context   = 1024;
    configured.prefill_chunk = 128;
    configured.speculative   = {};
    ninfer::test::qwen3_5::set_context_capacity(configured, 1);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 2;
    request.execution.sampling.temperature    = 0.0F;
    request.stop.include_model_defaults       = false;

    // Both inputs span several chunks. Raw input retains P; Chat retains its typed R.
    // Each fresh Engine has room for that one recovery state and its active writer.
    for (const bool chat : {false, true}) {
        ninfer::Engine engine(configured);
        ninfer::PreparedPrompt prepared;
        if (chat) {
            ninfer::PromptInput input;
            input.options.enable_thinking                              = true;
            input.options.preserve_thinking                            = true;
            input.context_cache.allow_engine_automatic_shared_prefixes = false;
            std::string text                                           = "A";
            for (std::uint32_t index = 1; index < 512; ++index) { text += " A"; }
            ninfer::ChatMessage user;
            user.role = ninfer::ChatRole::User;
            user.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
            input.messages.push_back(std::move(user));
            prepared = engine.prepare(std::move(input));
        } else {
            prepared = engine.prepare_tokens(std::vector<ninfer::TokenId>(512, 5834));
        }
        const auto result = engine.generate(std::move(prepared), request);
        const auto stats  = engine.snapshot().runtime;
        if (result.generated_token_ids.size() != 2 ||
            result.prompt.prompt_tokens <= configured.prefill_chunk ||
            result.stats.computed_prefill_tokens != result.prompt.prompt_tokens ||
            stats.active_captures_completed != 1 || stats.active_captures_aborted != 0 ||
            stats.context_transfers.state[0].images != 0) {
            std::cerr << (chat ? "Chat R" : "raw P")
                      << " did not remain the only prefill capture: prompt="
                      << result.prompt.prompt_tokens
                      << " computed=" << result.stats.computed_prefill_tokens
                      << " captures=" << stats.active_captures_completed
                      << " aborted=" << stats.active_captures_aborted
                      << " state_d2h=" << stats.context_transfers.state[0].images << '\n';
            return 1;
        }
    }
    return 0;
}

int exercise_host_restore(const char* artifact) {
    ninfer::Engine engine(host_restore_engine_options(artifact));
    auto options = [](std::uint32_t outputs, bool reuse) {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = outputs;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.allow_prefix_reuse      = reuse;
        request.stop.include_model_defaults       = false;
        return request;
    };

    // One full prefill chunk leaves a prompt and a later generation checkpoint on Device.
    // A token branch isolates typed Host restore from template rewriting and
    // the retention of a shorter, optional chat rewrite anchor.
    const std::vector<ninfer::TokenId> retained_input(256, 5834);
    const ninfer::GenerationResult retained =
        engine.generate(engine.prepare_tokens(retained_input), options(5, true));
    if (retained.prompt.prompt_tokens != 256 || retained.generated_token_ids.size() != 5) {
        std::cerr << "Host-restore source request did not complete\n";
        return 1;
    }

    auto continuation = retained_input;
    // Match only the prompt checkpoint: a deeper generation point may retain Device state.
    continuation.push_back(retained.generated_token_ids.front() == 198 ? 5834 : 198);
    continuation.insert(continuation.end(), 5, 198);

    const ninfer::RuntimeStats before_pressure = engine.snapshot().runtime;
    const ninfer::GenerationResult pressure_result =
        engine.generate(engine.prepare_tokens(continuation), options(2, false));
    const ninfer::RuntimeStats after_pressure = engine.snapshot().runtime;
    const auto transfer_bytes = [](const ninfer::RuntimeStats& stats, std::size_t direction) {
        auto bytes = stats.context_transfers.state[direction].bytes;
        for (const auto& history : stats.context_transfers.kv) bytes += history[direction].bytes;
        return bytes;
    };
    const auto backed_up = transfer_bytes(after_pressure, 0) - transfer_bytes(before_pressure, 0);
    if (pressure_result.generated_token_ids.size() != 2 || backed_up == 0) {
        std::cerr << "Host pressure did not demote checkpoint records\n";
        return 1;
    }

    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare_tokens(continuation), options(2, true));
    const ninfer::RuntimeStats after_restore = engine.snapshot().runtime;
    const auto restored_bytes =
        transfer_bytes(after_restore, 1) - transfer_bytes(after_pressure, 1);
    if (restored.generated_token_ids.size() != 2 ||
        restored.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        restored.stats.reused_prompt_tokens != retained_input.size() || restored_bytes == 0 ||
        restored_bytes > transfer_bytes(after_pressure, 0)) {
        std::cerr << "MTP checkpoint did not restore its missing prefix replicas: path="
                  << static_cast<int>(restored.stats.prefix_reuse_path)
                  << " reused=" << restored.stats.reused_prompt_tokens
                  << " outputs=" << restored.generated_token_ids.size()
                  << " backed_up=" << backed_up << " restored=" << restored_bytes << '\n';
        return 1;
    }
    // State and each history now compete for the same bytes. Only missing replicas are restored;
    // this test does not require the eviction policy to move every record type to Host.
    for (std::size_t pool = 0; pool < after_restore.context_transfers.kv.size(); ++pool) {
        if (after_restore.context_transfers.kv[pool][1].page_groups -
                after_pressure.context_transfers.kv[pool][1].page_groups >
            after_pressure.context_transfers.kv[pool][0].page_groups) {
            std::cerr << "checkpoint restored more pages than its Host history contains\n";
            return 1;
        }
    }

    // The uncached pressure request and checkpoint resume use different valid prefill splits, so
    // the pressure result is a completion and transfer trigger rather than an exact-token oracle.
    return 0;
}

int exercise_explicit_prefix(const char* artifact) {
    auto options                              = engine_options(artifact);
    options.enable_vision                     = false;
    options.context_cache.host_capacity_bytes = 0;
    ninfer::Engine engine(std::move(options));
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 3;
    request.execution.sampling.temperature    = 0.0F;

    std::string description;
    for (std::uint32_t index = 0; index < 120; ++index) { description += "stable-schema "; }
    const std::string tool =
        std::string(R"({"type":"function","function":{"name":"lookup","description":")") +
        description +
        R"(","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})";
    const auto input = [&](std::string question, bool marker) {
        ninfer::PromptInput prompt;
        prompt.options.enable_thinking = false;
        prompt.options.tool_jsons.push_back(tool);
        prompt.context_cache.allow_engine_automatic_shared_prefixes = false;
        if (marker) {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
                .after_tool_count = 1,
            });
        }
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
        prompt.messages.push_back(std::move(user));
        return prompt;
    };
    request.execution.allow_prefix_reuse = true;
    const auto source =
        engine.generate(engine.prepare(input("Use lookup for alpha.", true)), request);
    const auto branch =
        engine.generate(engine.prepare(input("Use lookup for bravo.", false)), request);
    request.execution.allow_prefix_reuse = false;
    const auto cold =
        engine.generate(engine.prepare(input("Use lookup for bravo.", false)), request);
    if (source.generated_token_ids.size() != 3 || branch.generated_token_ids.size() != 3 ||
        cold.generated_token_ids.size() != 3 ||
        branch.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        branch.stats.reused_prompt_tokens == 0 ||
        branch.stats.reused_prompt_tokens >= branch.prompt.prompt_tokens ||
        cold.stats.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        cold.stats.reused_prompt_tokens != 0) {
        std::cerr
            << "explicit tool prefix was not independently reusable by a changed user suffix\n";
        return 1;
    }
    return 0;
}

std::string nested_tool_definition(std::string name, std::string word) {
    std::string description;
    for (std::uint32_t index = 0; index < 160; ++index) {
        description += word;
        description.push_back(' ');
    }
    return std::string(R"({"type":"function","function":{"name":")") + name +
           R"(","description":")" + description +
           R"(","parameters":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]}}})";
}

ninfer::PromptInput nested_tool_prompt(std::vector<std::string> tools, std::string question) {
    ninfer::PromptInput input;
    input.options.enable_thinking                              = false;
    input.options.tool_jsons                                   = std::move(tools);
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    for (std::uint32_t count = 1; count <= input.options.tool_jsons.size(); ++count) {
        input.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = count,
        });
    }
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
    input.messages.push_back(std::move(user));
    return input;
}

int exercise_nested_tool_markers(const char* artifact) {
    auto configured = anthropic_prefix_regression_engine_options(artifact);
    // Keep both shared markers and the private response/end points alongside active work.
    configured.context_cache.host_capacity_bytes = 512ULL << 20;
    ninfer::test::qwen3_5::set_context_capacity(configured, 4);
    ninfer::Engine engine(std::move(configured));
    const std::string alpha   = nested_tool_definition("alpha", "stable-alpha");
    const std::string bravo   = nested_tool_definition("bravo", "stable-bravo");
    const std::string charlie = nested_tool_definition("charlie", "branch-charlie");
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    // A single source creates both markers. Each probe changes the user suffix so the
    // private response/endpoint cannot satisfy the shared-prefix conformance check.
    const auto seed = engine.generate(
        engine.prepare(nested_tool_prompt({alpha, bravo}, "Use one listed function.")), request);
    const auto before_deep = engine.snapshot().runtime;
    const auto deep        = engine.generate(
        engine.prepare(nested_tool_prompt({alpha, bravo}, "Choose a function for the next task.")),
        request);
    const auto after_deep = engine.snapshot().runtime;
    const auto shallow    = engine.generate(
        engine.prepare(nested_tool_prompt({alpha, charlie}, "Select a function for this branch.")),
        request);
    const auto after_shallow = engine.snapshot().runtime;
    const auto accounted     = [](const ninfer::GenerationResult& result) {
        return result.generated_token_ids.size() == 1 &&
               result.finish_reason == ninfer::FinishReason::OutputLimit &&
               result.stats.reused_prompt_tokens <= result.prompt.prompt_tokens &&
               result.stats.computed_prefill_tokens ==
                   result.prompt.prompt_tokens - result.stats.reused_prompt_tokens;
    };
    if (!accounted(seed) || !accounted(deep) || !accounted(shallow) ||
        seed.stats.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        seed.stats.reused_prompt_tokens != 0 ||
        deep.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        shallow.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        shallow.stats.reused_prompt_tokens == 0 ||
        deep.stats.reused_prompt_tokens <= shallow.stats.reused_prompt_tokens ||
        deep.stats.reused_prompt_tokens >= deep.prompt.prompt_tokens ||
        shallow.stats.reused_prompt_tokens >= shallow.prompt.prompt_tokens ||
        after_deep.computed_prefill_tokens - before_deep.computed_prefill_tokens !=
            deep.stats.computed_prefill_tokens ||
        after_shallow.computed_prefill_tokens - after_deep.computed_prefill_tokens !=
            shallow.stats.computed_prefill_tokens) {
        std::cerr << "one source did not establish both nested tool markers: deep_reused="
                  << deep.stats.reused_prompt_tokens << " deep_prompt=" << deep.prompt.prompt_tokens
                  << " deep_computed=" << deep.stats.computed_prefill_tokens
                  << " shallow_reused=" << shallow.stats.reused_prompt_tokens
                  << " shallow_prompt=" << shallow.prompt.prompt_tokens
                  << " shallow_computed=" << shallow.stats.computed_prefill_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_anthropic_prefix_regression(const char* artifact) {
    ninfer::Engine engine(anthropic_prefix_regression_engine_options(artifact));
    const auto conversation = [](bool followup, const ninfer::GenerationResult* first = nullptr) {
        ninfer::PromptInput input;
        input.options.enable_thinking   = true;
        input.options.preserve_thinking = true;
        input.context_cache.session_key = "anthropic-prefix-regression";

        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                 .text  = "Reply with exactly the word blue.",
                                                 .media = {}});
        input.messages.push_back(std::move(user));
        if (!followup) { return input; }
        if (first == nullptr) { throw std::logic_error("followup fixture has no source result"); }

        ninfer::ChatMessage assistant;
        assistant.role              = ninfer::ChatRole::Assistant;
        assistant.reasoning_content = first->reasoning;
        if (!first->content.empty()) {
            assistant.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = first->content, .media = {}});
        }
        input.messages.push_back(std::move(assistant));
        ninfer::ChatMessage next;
        next.role = ninfer::ChatRole::User;
        next.parts.push_back(ninfer::MessagePart{.kind  = ninfer::MessagePartKind::Text,
                                                 .text  = "Now reply with exactly the word green.",
                                                 .media = {}});
        input.messages.push_back(std::move(next));
        return input;
    };
    const auto generation_options = [](std::uint32_t outputs, bool model_stops) {
        ninfer::RequestOptions options;
        options.execution.requested_output_tokens = outputs;
        options.execution.sampling.temperature    = 0.0F;
        options.execution.allow_prefix_reuse      = true;
        options.stop.include_model_defaults       = model_stops;
        return options;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(conversation(false)), generation_options(192, true));
    if (first.reasoning.empty() || first.content.empty() ||
        first.finish_reason != ninfer::FinishReason::StopToken) {
        std::cerr << "endpoint regression fixture did not produce a closed reasoning response: "
                  << "reasoning=" << first.reasoning.size() << " content=" << first.content.size()
                  << " finish=" << static_cast<int>(first.finish_reason) << '\n';
        return 1;
    }
    const ninfer::RuntimeStats before_followup = engine.snapshot().runtime;
    const ninfer::GenerationResult followup =
        engine.generate(engine.prepare(conversation(true, &first)), generation_options(1, false));
    const ninfer::RuntimeStats after_followup = engine.snapshot().runtime;
    const std::uint64_t followup_prefill =
        after_followup.computed_prefill_tokens - before_followup.computed_prefill_tokens;
    if (followup.generated_token_ids.size() != 1 ||
        followup.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        followup.stats.reused_prompt_tokens == 0 ||
        followup_prefill != followup.prompt.prompt_tokens - followup.stats.reused_prompt_tokens) {
        std::cerr << "model-output reasoning frontier did not resume from its checkpoint: path="
                  << static_cast<int>(followup.stats.prefix_reuse_path)
                  << " reused=" << followup.stats.reused_prompt_tokens
                  << " prompt=" << followup.prompt.prompt_tokens << " computed=" << followup_prefill
                  << '\n';
        return 1;
    }

    const std::string alpha   = nested_tool_definition("alpha", "stable-alpha");
    const std::string bravo   = nested_tool_definition("bravo", "stable-bravo");
    const std::string charlie = nested_tool_definition("charlie", "branch-charlie");
    const auto filler_prompt  = [](std::string text) {
        ninfer::PromptInput input;
        input.options.enable_thinking                              = false;
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(ninfer::MessagePart{
             .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        input.messages.push_back(std::move(user));
        return input;
    };

    const ninfer::RequestOptions one_token = generation_options(1, false);
    const ninfer::GenerationResult seed    = engine.generate(
        engine.prepare(nested_tool_prompt({alpha, bravo}, "Use one listed function.")), one_token);
    const ninfer::GenerationResult first_filler = engine.generate(
        engine.prepare(filler_prompt("Replace the private seed continuation.")), one_token);
    if (seed.generated_token_ids.size() != 1 || first_filler.generated_token_ids.size() != 1) {
        std::cerr << "shared compact fixture did not establish its seed and filler\n";
        return 1;
    }

    const ninfer::RuntimeStats before_late = engine.snapshot().runtime;
    const ninfer::GenerationResult late    = engine.generate(
        engine.prepare(nested_tool_prompt({alpha, bravo}, "Use one listed function.")), one_token);
    const ninfer::RuntimeStats after_late = engine.snapshot().runtime;
    const std::uint64_t late_prefill =
        after_late.computed_prefill_tokens - before_late.computed_prefill_tokens;
    const ninfer::GenerationResult second_filler = engine.generate(
        engine.prepare(filler_prompt("Replace the private late continuation.")), one_token);
    const ninfer::RuntimeStats before_branch = engine.snapshot().runtime;
    const ninfer::GenerationResult branch    = engine.generate(
        engine.prepare(nested_tool_prompt({alpha, charlie}, "Use one listed function.")),
        one_token);
    const ninfer::RuntimeStats after_branch = engine.snapshot().runtime;
    const std::uint64_t branch_prefill =
        after_branch.computed_prefill_tokens - before_branch.computed_prefill_tokens;

    const auto host =
        ninfer::test::qwen3_5::context_usage(after_branch, ninfer::ContextMemoryDomain::Host);
    // This sequence also retains the earlier reasoning dialogue and unrelated private fillers.
    // Report optional-prefix eviction under that fixed pressure rather than assuming all of
    // those histories fit. The independent nested-tool-markers scenario verifies both captures.
    std::cout << "anthropic mixed pressure: late_reused=" << late.stats.reused_prompt_tokens
              << " late_prompt=" << late.prompt.prompt_tokens << " late_computed=" << late_prefill
              << " late_ttft_ms=" << late.stats.timings.first_token_seconds * 1000
              << " branch_reused=" << branch.stats.reused_prompt_tokens
              << " branch_prompt=" << branch.prompt.prompt_tokens
              << " branch_computed=" << branch_prefill
              << " branch_ttft_ms=" << branch.stats.timings.first_token_seconds * 1000
              << " host_peak_bytes=" << host.peak_occupied_bytes
              << " host_capacity_bytes=" << host.capacity_bytes << '\n';
    const auto accounted = [](const ninfer::GenerationResult& result) {
        return result.generated_token_ids.size() == 1 &&
               result.finish_reason == ninfer::FinishReason::OutputLimit &&
               result.stats.reused_prompt_tokens <= result.prompt.prompt_tokens &&
               result.stats.computed_prefill_tokens ==
                   result.prompt.prompt_tokens - result.stats.reused_prompt_tokens;
    };
    if (!accounted(seed) || !accounted(first_filler) || !accounted(late) ||
        !accounted(second_filler) || !accounted(branch) ||
        late_prefill != late.stats.computed_prefill_tokens ||
        branch_prefill != branch.stats.computed_prefill_tokens ||
        ninfer::test::qwen3_5::occupied_bytes(host) > host.capacity_bytes ||
        host.peak_occupied_bytes > host.capacity_bytes) {
        std::cerr << "mixed-prefix pressure violated request or physical Host accounting\n";
        return 1;
    }
    return 0;
}

int exercise_shared_rewrite_materialization(const char* artifact) {
    ninfer::Engine engine(shared_rewrite_materialization_engine_options(artifact));

    std::vector<std::string> tools;
    tools.reserve(40);
    tools.push_back(
        R"({"type":"function","function":{"name":"read_chunk","description":"Read the next diagnostic chunk. Always use this tool until told done.","parameters":{"type":"object","properties":{"chunk":{"type":"integer"}},"required":["chunk"]}}})");
    for (std::uint32_t index = 1; index < 40; ++index) {
        tools.push_back(
            std::string(R"({"type":"function","function":{"name":"unused_tool_)") +
            std::to_string(index) +
            R"(","description":"Unused diagnostic tool.","parameters":{"type":"object","properties":{"value":{"type":"string"}}}}})");
    }

    constexpr std::string_view system_text =
        "You are testing a tool loop. On every turn call read_chunk exactly once with the next "
        "integer chunk number. Do not finish or answer in prose.";
    std::vector<ninfer::ChatMessage> messages;
    ninfer::ChatMessage system;
    system.role = ninfer::ChatRole::System;
    system.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::string(system_text), .media = {}});
    messages.push_back(std::move(system));
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind  = ninfer::MessagePartKind::Text,
        .text  = "Start by calling read_chunk with chunk 1.",
        .media = {},
    });
    messages.push_back(std::move(user));

    const auto input = [&](bool latest_is_tool) {
        ninfer::PromptInput prompt;
        prompt.messages                  = messages;
        prompt.options.enable_thinking   = true;
        prompt.options.preserve_thinking = true;
        prompt.options.tool_jsons        = tools;
        prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
            .leading_instruction_bytes = static_cast<std::uint32_t>(system_text.size()),
        });
        prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
            .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = static_cast<std::uint32_t>(tools.size()),
        });
        if (latest_is_tool) {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count = static_cast<std::uint32_t>(messages.size()),
                .kind                = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence            = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
            });
        } else {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count      = static_cast<std::uint32_t>(messages.size()),
                .kind                     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence                 = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
                .after_message_part_count = 1,
            });
        }
        return prompt;
    };
    const auto request_options = [] {
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = 16384;
        request.execution.sampling.temperature    = 0.0F;
        request.execution.thinking.budget         = 1024;
        request.execution.allow_prefix_reuse      = true;
        return request;
    };
    const auto append_result = [&](const ninfer::GenerationResult& result, std::uint32_t turn) {
        if (result.tool_calls.size() != 1) {
            throw std::logic_error("shared rewrite fixture did not produce one tool call");
        }
        ninfer::ChatMessage assistant;
        assistant.role              = ninfer::ChatRole::Assistant;
        assistant.reasoning_content = result.reasoning + "\nHistory normalized by the client.";
        if (!result.content.empty()) {
            assistant.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = result.content, .media = {}});
        }
        const std::string call_id = "call_" + std::to_string(turn);
        assistant.tool_calls.push_back(ninfer::ToolCall{
            .id             = call_id,
            .name           = result.tool_calls.front().name,
            .arguments_json = result.tool_calls.front().arguments_json,
        });
        messages.push_back(std::move(assistant));

        std::string diagnostic;
        diagnostic.reserve(64000);
        for (std::uint32_t line = 0; line < 500; ++line) {
            diagnostic += "chunk=" + std::to_string(turn) + " line=" + std::to_string(line) +
                          " key=value abcdefghijklmnopqrstuvwxyz0123456789 "
                          "ABCDEFGHIJKLMNOPQRSTUVWXYZ9876543210\n";
        }
        ninfer::ChatMessage tool_result;
        tool_result.role         = ninfer::ChatRole::Tool;
        tool_result.tool_call_id = call_id;
        tool_result.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(diagnostic), .media = {}});
        messages.push_back(std::move(tool_result));
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input(false)), request_options());
    append_result(first, 1);
    const ninfer::GenerationResult second =
        engine.generate(engine.prepare(input(true)), request_options());
    append_result(second, 2);
    const ninfer::RuntimeStats before_third = engine.snapshot().runtime;
    const ninfer::GenerationResult third =
        engine.generate(engine.prepare(input(true)), request_options());
    const ninfer::RuntimeStats after_third = engine.snapshot().runtime;

    // With preserved thinking, R precedes the current assistant opener. The explicit marker
    // after the final tool message aliases this position. Normalizing the generated reasoning
    // invalidates E, so turn three must recover the R established by turn two, beyond the
    // system/tools prefix and the entire first tool result.
    const auto opener_tokens =
        static_cast<std::uint32_t>(engine.tokenize_text("<|im_start|>assistant\n<think>\n").size());
    const auto expected_reuse = second.prompt.prompt_tokens - opener_tokens;
    const auto third_prefill =
        after_third.computed_prefill_tokens - before_third.computed_prefill_tokens;

    if (first.stats.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        second.stats.reused_prompt_tokens == 0 ||
        third.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        third.stats.reused_prompt_tokens != expected_reuse ||
        third.stats.reused_prompt_tokens <= second.stats.reused_prompt_tokens ||
        third.stats.computed_prefill_tokens != third.prompt.prompt_tokens - expected_reuse ||
        third_prefill != third.stats.computed_prefill_tokens || third.generated_token_ids.empty() ||
        after_third.materialization_state_forks <= before_third.materialization_state_forks) {
        std::cerr << "normalized tool history did not preserve the advancing rewrite alias: "
                  << "first_path=" << static_cast<int>(first.stats.prefix_reuse_path)
                  << " second_path=" << static_cast<int>(second.stats.prefix_reuse_path)
                  << " second_reused=" << second.stats.reused_prompt_tokens
                  << " third_path=" << static_cast<int>(third.stats.prefix_reuse_path)
                  << " third_reused=" << third.stats.reused_prompt_tokens
                  << " expected_reused=" << expected_reuse
                  << " third_prompt=" << third.prompt.prompt_tokens
                  << " third_computed=" << third.stats.computed_prefill_tokens
                  << " interval_computed=" << third_prefill
                  << " third_outputs=" << third.generated_token_ids.size()
                  << " materialization_forks=" << before_third.materialization_state_forks << '/'
                  << after_third.materialization_state_forks << '\n';
        return 1;
    }
    return 0;
}

int exercise_explicit_anchor_branch(const char* artifact) {
    ninfer::Engine engine(explicit_anchor_engine_options(artifact));

    const auto input = [](std::vector<std::string> turns,
                          std::optional<std::uint32_t> marker_after) {
        ninfer::PromptInput prompt;
        for (std::string& text : turns) {
            ninfer::ChatMessage message;
            message.role = ninfer::ChatRole::User;
            message.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
            prompt.messages.push_back(std::move(message));
        }
        prompt.options.enable_thinking = false;
        if (marker_after) {
            prompt.context_cache.markers.push_back(ninfer::PromptCacheMarker{
                .after_message_count = *marker_after,
                .kind                = ninfer::PromptCacheMarkerKind::PrivateLongAnchor,
                .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
            });
        }
        return prompt;
    };
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 1;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;

    constexpr std::string_view stable =
        "This is the stable conversation prefix retained for a later branch.";
    const ninfer::GenerationResult source = engine.generate(
        engine.prepare(input({std::string(stable), "Follow the original branch."}, 1)), request);
    if (source.generated_token_ids.size() != 1 ||
        source.stats.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "first explicit intermediate checkpoint capture did not complete from Root\n";
        return 1;
    }

    ninfer::PromptInput replacement_input =
        input({std::string(stable), "Follow the replacement branch.",
               "This suffix belongs only to the replacement source."},
              2);
    // Keep the session hint while changing the final suffix; matching still uses the input
    // identity.
    replacement_input.context_cache.session_key = "private-long-anchor-replacement";
    const ninfer::GenerationResult replacement =
        engine.generate(engine.prepare(std::move(replacement_input)), request);
    if (replacement.generated_token_ids.size() != 1 ||
        replacement.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        replacement.stats.reused_prompt_tokens == 0 ||
        replacement.stats.reused_prompt_tokens >= replacement.prompt.prompt_tokens) {
        std::cerr << "explicit intermediate checkpoint was not selected for a changed branch: path="
                  << static_cast<int>(replacement.stats.prefix_reuse_path)
                  << " reused=" << replacement.stats.reused_prompt_tokens
                  << " prompt=" << replacement.prompt.prompt_tokens << '\n';
        return 1;
    }

    ninfer::PromptInput replaced_input =
        input({std::string(stable), "Follow the replacement branch.",
               "Continue through a different branch suffix."},
              std::nullopt);
    replaced_input.context_cache.session_key = "private-long-anchor-replacement";
    const ninfer::GenerationResult replaced =
        engine.generate(engine.prepare(std::move(replaced_input)), request);
    if (replaced.generated_token_ids.size() != 1 ||
        replaced.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        replaced.stats.reused_prompt_tokens <= replacement.stats.reused_prompt_tokens ||
        replaced.stats.reused_prompt_tokens >= replaced.prompt.prompt_tokens) {
        const ninfer::RuntimeStats stats = engine.snapshot().runtime;
        std::cerr << "replacement explicit intermediate checkpoint was not reusable: path="
                  << static_cast<int>(replaced.stats.prefix_reuse_path)
                  << " first_reused=" << replacement.stats.reused_prompt_tokens
                  << " replaced_reused=" << replaced.stats.reused_prompt_tokens
                  << " prompt=" << replaced.prompt.prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    return 0;
}

int exercise_rewrite_checkpoints(ninfer::Engine& engine) {

    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    auto assistant_call = [&](std::string reasoning, std::string id, std::string key) {
        ninfer::ChatMessage message = text_message(ninfer::ChatRole::Assistant, "");
        message.reasoning_content   = std::move(reasoning);
        message.tool_calls.push_back(ninfer::ToolCall{
            .id = std::move(id), .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        return message;
    };
    auto input_with_history = [&](int completed_responses, bool preserve_thinking) {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (completed_responses >= 1) {
            input.messages.push_back(
                assistant_call("The first lookup should be alpha.", "call_alpha", "alpha"));
            ninfer::ChatMessage tool =
                text_message(ninfer::ChatRole::Tool, "{\"value\":17,\"next\":\"beta\"}");
            tool.tool_call_id = "call_alpha";
            input.messages.push_back(std::move(tool));
        }
        if (completed_responses >= 2) {
            input.messages.push_back(
                assistant_call("The alpha result requests beta.", "call_beta", "beta"));
            ninfer::ChatMessage tool = text_message(ninfer::ChatRole::Tool, "{\"value\":25}");
            tool.tool_call_id        = "call_beta";
            input.messages.push_back(std::move(tool));
        }
        input.options.preserve_thinking = preserve_thinking;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return input;
    };
    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 4;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input_with_history(0, true)), options(true));
    if (first.generated_token_ids.size() != 4 ||
        first.stats.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "response-checkpoint source request did not complete from a cold lane\n";
        return 1;
    }

    const ninfer::GenerationResult exact_replay =
        engine.generate(engine.prepare(input_with_history(0, false)), options(true));
    if (exact_replay.generated_token_ids.size() != 4 ||
        exact_replay.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        exact_replay.stats.reused_prompt_tokens == 0) {
        const ninfer::RuntimeStats stats = engine.snapshot().runtime;
        std::cerr << "pre-generation response checkpoint was not restored on an exact replay: "
                  << "path=" << static_cast<int>(exact_replay.stats.prefix_reuse_path)
                  << " reused=" << exact_replay.stats.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }
    const ninfer::GenerationResult exact_baseline =
        engine.generate(engine.prepare(input_with_history(0, false)), options(false));
    // Replay can rebuild the MTP bridge at T=1; capture can also split the source prefill.
    // Each route must finish within its own budget and publish a valid reuse frontier.
    if (exact_baseline.generated_token_ids.size() != 4 ||
        exact_baseline.stats.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        exact_baseline.stats.reused_prompt_tokens != 0) {
        std::cerr << "uncached response-checkpoint baseline did not complete from Root: path="
                  << static_cast<int>(exact_baseline.stats.prefix_reuse_path)
                  << " reused=" << exact_baseline.stats.reused_prompt_tokens
                  << " outputs=" << exact_baseline.generated_token_ids.size() << '\n';
        return 1;
    }

    const ninfer::GenerationResult first_replay =
        engine.generate(engine.prepare(input_with_history(1, true)), options(true));
    if (first_replay.generated_token_ids.size() != 4 ||
        first_replay.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        first_replay.stats.reused_prompt_tokens == 0) {
        std::cerr << "normalized tool response did not reuse a compatible checkpoint\n";
        return 1;
    }

    const ninfer::GenerationResult second_replay =
        engine.generate(engine.prepare(input_with_history(2, true)), options(true));
    if (second_replay.generated_token_ids.size() != 4 ||
        second_replay.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        second_replay.stats.reused_prompt_tokens <= first_replay.stats.reused_prompt_tokens) {
        const ninfer::RuntimeStats stats = engine.snapshot().runtime;
        std::cerr << "rolling response checkpoint did not advance across the tool loop: first="
                  << first_replay.stats.reused_prompt_tokens
                  << " second=" << second_replay.stats.reused_prompt_tokens
                  << " captures=" << stats.active_captures_completed
                  << " capture_aborts=" << stats.active_captures_aborted << '\n';
        return 1;
    }

    const ninfer::GenerationResult mode_change =
        engine.generate(engine.prepare(input_with_history(2, false)), options(true));
    if (mode_change.generated_token_ids.size() != 4 ||
        mode_change.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        mode_change.stats.reused_prompt_tokens == 0) {
        std::cerr << "preserve-thinking policy change discarded a compatible response checkpoint: "
                  << "path=" << static_cast<int>(mode_change.stats.prefix_reuse_path)
                  << " reused=" << mode_change.stats.reused_prompt_tokens << '\n';
        return 1;
    }

    return 0;
}

int exercise_agent_continuation(const char* artifact) {
    auto configured = anthropic_prefix_regression_engine_options(artifact);
    // Keep the main and branch response/end points alongside active work; this trajectory
    // exercises ownership without intentionally forcing eviction.
    configured.max_context                       = 4096;
    configured.context_cache.host_capacity_bytes = 0;
    ninfer::test::qwen3_5::set_context_capacity(configured, 4);
    ninfer::Engine engine(std::move(configured));
    const auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    ninfer::PromptInput input;
    input.options.enable_thinking                              = true;
    input.options.preserve_thinking                            = true;
    input.context_cache.session_key                            = "agent-continuation-main";
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    input.options.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"step":{"type":"integer"}},"required":["step"]}}})");
    input.messages.push_back(text_message(
        ninfer::ChatRole::User, "Use the lookup results to determine the next diagnostic step."));
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 4;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;
    const auto opener_tokens =
        static_cast<std::uint32_t>(engine.tokenize_text("<|im_start|>assistant\n<think>\n").size());
    const auto closing_tokens =
        static_cast<std::uint32_t>(engine.tokenize_text("<|im_end|>\n").size());
    const auto archived_token    = engine.tokenize_text("Archived").front();
    std::uint64_t total_reused   = 0;
    std::uint64_t total_computed = 0;
    const auto generate          = [&](ninfer::PromptInput prompt, std::uint32_t expected_frontier,
                              std::uint32_t round,
                              std::string_view lineage) -> std::optional<ninfer::GenerationResult> {
        const auto before = engine.snapshot().runtime;
        auto result       = engine.generate(engine.prepare(std::move(prompt)), request);
        const auto after  = engine.snapshot().runtime;
        if (result.generated_token_ids.size() != 4 ||
            result.finish_reason != ninfer::FinishReason::OutputLimit ||
            result.stats.prefix_reuse_path != (expected_frontier
                                                            ? ninfer::PrefixReusePath::Checkpoint
                                                            : ninfer::PrefixReusePath::Root) ||
            result.stats.reused_prompt_tokens != expected_frontier ||
            result.stats.reused_prompt_tokens >= result.prompt.prompt_tokens ||
            result.stats.computed_prefill_tokens !=
                result.prompt.prompt_tokens - result.stats.reused_prompt_tokens ||
            after.computed_prefill_tokens - before.computed_prefill_tokens !=
                result.stats.computed_prefill_tokens) {
            std::cerr << "agent continuation " << lineage << " round=" << round
                      << " reused=" << result.stats.reused_prompt_tokens
                      << " expected=" << expected_frontier
                      << " prompt=" << result.prompt.prompt_tokens
                      << " computed=" << result.stats.computed_prefill_tokens
                      << " interval_computed="
                      << after.computed_prefill_tokens - before.computed_prefill_tokens
                      << " outputs=" << result.generated_token_ids.size() << '\n';
            return std::nullopt;
        }
        total_reused += result.stats.reused_prompt_tokens;
        total_computed += result.stats.computed_prefill_tokens;
        return result;
    };

    auto previous = generate(input, 0, 0, "main");
    if (!previous) { return 1; }
    constexpr std::uint32_t rounds = 16;
    for (std::uint32_t round = 1; round < rounds; ++round) {
        const auto prior_role       = input.messages.back().role;
        const bool content_recovery = prior_role == ninfer::ChatRole::User ||
                                      prior_role == ninfer::ChatRole::System ||
                                      prior_role == ninfer::ChatRole::Developer;
        const auto expected_frontier = previous->prompt.prompt_tokens - opener_tokens -
                                       (content_recovery ? closing_tokens : 0U);
        if (expected_frontier <= previous->stats.reused_prompt_tokens) {
            std::cerr << "agent continuation did not advance its input recovery position\n";
            return 1;
        }

        // As in the short typed-R fixture, the client supplies normalized structured tool
        // history. Choose a different first reasoning token so the generated E cannot match;
        // Every round must recover the previous input point, independent of the sampled output.
        auto assistant = text_message(ninfer::ChatRole::Assistant, "");
        assistant.reasoning_content =
            std::string(previous->generated_token_ids.front() == archived_token ? "Recorded"
                                                                                : "Archived") +
            " normalized lookup step " + std::to_string(round) + '.';
        const std::string call_id = "lookup_" + std::to_string(round);
        assistant.tool_calls.push_back(ninfer::ToolCall{
            .id             = call_id,
            .name           = "lookup",
            .arguments_json = "{\"step\":" + std::to_string(round) + '}',
        });
        input.messages.push_back(std::move(assistant));
        auto tool =
            text_message(ninfer::ChatRole::Tool, "{\"value\":" + std::to_string(round * 17) + '}');
        tool.tool_call_id = call_id;
        input.messages.push_back(std::move(tool));
        if (round == 5) {
            input.messages.push_back(text_message(
                ninfer::ChatRole::System, "Keep using the established diagnostic sequence."));
        } else if (round == 11) {
            input.messages.push_back(text_message(
                ninfer::ChatRole::Developer, "Preserve prior results and inspect the next step."));
        }

        if (round == 8) {
            auto branch                               = input;
            branch.context_cache.session_key          = "agent-continuation-branch";
            branch.messages.back().parts.front().text = "{\"value\":999,\"branch\":\"alternate\"}";
            const auto before_branch                  = engine.snapshot().runtime;
            if (!generate(std::move(branch), expected_frontier, round, "branch")) { return 1; }
            if (engine.snapshot().runtime.materialization_state_forks <=
                before_branch.materialization_state_forks) {
                std::cerr << "agent branch did not preserve its independent parent state\n";
                return 1;
            }
        }
        previous = generate(input, expected_frontier, round, "main");
        if (!previous) { return 1; }
    }

    const auto settled = engine.snapshot().runtime;
    const auto host =
        ninfer::test::qwen3_5::context_usage(settled, ninfer::ContextMemoryDomain::Host);
    const auto device =
        ninfer::test::qwen3_5::context_usage(settled, ninfer::ContextMemoryDomain::Device);
    if (settled.running_requests || settled.waiting_requests || settled.paused_requests ||
        settled.replaying_requests || settled.materializing_requests ||
        settled.prefilling_requests || settled.decode_ready_requests ||
        settled.capture_pending_requests || settled.terminal_pending_requests ||
        host.reserved_bytes || host.pending_retirement_bytes ||
        host.peak_occupied_bytes > host.capacity_bytes) {
        std::cerr << "agent continuation left active membership or transfer reservations: running="
                  << settled.running_requests << " waiting=" << settled.waiting_requests
                  << " paused=" << settled.paused_requests
                  << " replaying=" << settled.replaying_requests
                  << " materializing=" << settled.materializing_requests
                  << " prefilling=" << settled.prefilling_requests
                  << " decode=" << settled.decode_ready_requests
                  << " capture=" << settled.capture_pending_requests
                  << " terminal=" << settled.terminal_pending_requests
                  << " reserved=" << host.reserved_bytes << '\n';
        return 1;
    }
    std::cout << "agent continuation: main_rounds=" << rounds << " branches=1"
              << " final_reused=" << previous->stats.reused_prompt_tokens
              << " total_reused=" << total_reused << " total_computed=" << total_computed
              << " retained_device_bytes=" << device.live_bytes
              << " retained_state_objects=" << settled.state_objects
              << " retained_main_pages=" << settled.kv_pools[0].occupied_page_groups << '\n';
    return 0;
}

int exercise_rewrite_branch(const char* artifact) {
    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    const auto input = [&](bool branch) {
        ninfer::PromptInput value;
        value.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (branch) {
            value.messages.push_back(text_message(ninfer::ChatRole::User,
                                                  "Summarize the conversation before answering."));
        }
        value.options.preserve_thinking = true;
        value.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return value;
    };
    const auto options = [](bool reuse) {
        ninfer::RequestOptions value;
        value.execution.requested_output_tokens = 4;
        value.execution.sampling.temperature    = 0.0F;
        value.execution.allow_prefix_reuse      = reuse;
        value.stop.include_model_defaults       = false;
        return value;
    };

    auto configured = engine_options(artifact);
    ninfer::test::qwen3_5::set_context_capacity(configured, 2);
    ninfer::Engine engine(std::move(configured));
    const ninfer::GenerationResult source =
        engine.generate(engine.prepare(input(false)), options(true));
    if (source.generated_token_ids.size() != 4 ||
        source.stats.prefix_reuse_path != ninfer::PrefixReusePath::Root) {
        std::cerr << "rewrite branch source did not establish a response checkpoint\n";
        return 1;
    }
    const ninfer::GenerationResult branch =
        engine.generate(engine.prepare(input(true)), options(true));
    const ninfer::GenerationResult branch_baseline =
        engine.generate(engine.prepare(input(true)), options(false));
    if (branch.generated_token_ids.size() != 4 || branch.stats.reused_prompt_tokens == 0 ||
        branch.stats.reused_prompt_tokens >= branch.prompt.prompt_tokens ||
        branch.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
        branch_baseline.generated_token_ids.size() != 4 ||
        branch_baseline.stats.reused_prompt_tokens != 0) {
        std::cerr << "replacement user suffix did not reuse the stable conversation prefix: path="
                  << static_cast<int>(branch.stats.prefix_reuse_path)
                  << " reused=" << branch.stats.reused_prompt_tokens
                  << " prompt=" << branch.prompt.prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_late_instructions(const char* artifact) {
    auto configured = explicit_anchor_engine_options(artifact);
    ninfer::test::qwen3_5::set_context_capacity(configured);
    ninfer::Engine engine(std::move(configured));
    const auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    ninfer::PromptInput input;
    input.options.enable_thinking                              = false;
    input.options.preserve_thinking                            = true;
    input.context_cache.allow_engine_automatic_shared_prefixes = false;
    input.messages.push_back(
        text_message(ninfer::ChatRole::System, "Explain engineering concepts in clear prose."));
    input.messages.push_back(text_message(
        ninfer::ChatRole::User,
        "Write a long paragraph explaining how a computer executes a sequence of instructions."));
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 16;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = true;
    request.stop.include_model_defaults       = false;
    auto previous                             = engine.generate(engine.prepare(input), request);
    if (previous.generated_token_ids.size() != 16 ||
        previous.stats.prefix_reuse_path != ninfer::PrefixReusePath::Root ||
        previous.content.empty()) {
        std::cerr << "late-instruction source did not produce a reconstructible response\n";
        return 1;
    }

    // Both official templates emit this opener when thinking is disabled. The previous
    // user body is the retained state, before the closing tokens and this opener; rebuilding an
    // output may require that point when its deeper generation endpoint no longer matches the
    // template's serialization.
    const auto recovery_suffix_tokens = static_cast<std::uint32_t>(
        engine.tokenize_text("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n").size());
    for (const auto role : {ninfer::ChatRole::System, ninfer::ChatRole::Developer}) {
        auto assistant              = text_message(ninfer::ChatRole::Assistant, previous.content);
        assistant.reasoning_content = previous.reasoning;
        input.messages.push_back(std::move(assistant));
        input.messages.push_back(
            text_message(role, role == ninfer::ChatRole::System
                                   ? "For the next answer, emphasize concrete examples."
                                   : "Keep the next explanation technical and practical."));
        input.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Continue with another long paragraph describing the next stage of execution."));

        const auto required_frontier = previous.prompt.prompt_tokens - recovery_suffix_tokens;
        const auto before            = engine.snapshot().runtime;
        auto next                    = engine.generate(engine.prepare(input), request);
        const auto after             = engine.snapshot().runtime;
        if (next.generated_token_ids.size() != 16 || next.content.empty() ||
            next.finish_reason != ninfer::FinishReason::OutputLimit ||
            next.stats.prefix_reuse_path != ninfer::PrefixReusePath::Checkpoint ||
            next.stats.reused_prompt_tokens < required_frontier ||
            next.stats.reused_prompt_tokens <= previous.stats.reused_prompt_tokens ||
            next.stats.computed_prefill_tokens !=
                next.prompt.prompt_tokens - next.stats.reused_prompt_tokens ||
            after.computed_prefill_tokens - before.computed_prefill_tokens !=
                next.stats.computed_prefill_tokens) {
            std::cerr << "late " << (role == ninfer::ChatRole::System ? "system" : "developer")
                      << " instruction invalidated the stable conversation prefix: reused="
                      << next.stats.reused_prompt_tokens << " required=" << required_frontier
                      << " prior_reused=" << previous.stats.reused_prompt_tokens
                      << " prompt=" << next.prompt.prompt_tokens
                      << " computed=" << next.stats.computed_prefill_tokens << '\n';
            return 1;
        }
        previous = std::move(next);
    }
    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    const auto image_bytes = gradient_ppm();
    auto image_part        = [](const std::vector<std::uint8_t>& bytes, std::string name) {
        ninfer::MessagePart image;
        image.kind              = ninfer::MessagePartKind::Media;
        image.media.kind        = ninfer::MediaKind::Image;
        image.media.bytes       = bytes;
        image.media.media_type  = "image/x-portable-pixmap";
        image.media.source_name = std::move(name);
        return image;
    };
    auto assistant_message = [](const ninfer::GenerationResult& result) {
        ninfer::ChatMessage message;
        message.role              = ninfer::ChatRole::Assistant;
        message.reasoning_content = result.reasoning;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = result.content, .media = {}});
        return message;
    };
    auto first_input = [&](const std::vector<std::uint8_t>& bytes) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(image_part(bytes, "inline.ppm"));
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking   = false;
        input.context_cache.session_key = "vision-prefix-real";
        return input;
    };
    auto followup_input = [&](const std::vector<std::uint8_t>& bytes,
                              const ninfer::GenerationResult& first) {
        ninfer::PromptInput input = first_input(bytes);
        input.messages.push_back(assistant_message(first));
        ninfer::ChatMessage followup;
        followup.role = ninfer::ChatRole::User;
        followup.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "Give one more detail.", .media = {}});
        input.messages.push_back(std::move(followup));
        return input;
    };
    auto appended_media_input =
        [&](const std::vector<std::uint8_t>& old_bytes, const ninfer::GenerationResult& first,
            const ninfer::GenerationResult& second, const std::vector<std::uint8_t>& new_bytes) {
            ninfer::PromptInput input = followup_input(old_bytes, first);
            input.messages.push_back(assistant_message(second));
            ninfer::ChatMessage followup;
            followup.role = ninfer::ChatRole::User;
            followup.parts.push_back(image_part(new_bytes, "second.ppm"));
            followup.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = "Compare the images.", .media = {}});
            input.messages.push_back(std::move(followup));
            return input;
        };

    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 2;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    // The 1024 merged Vision columns begin after the chat prefix, so the same item necessarily
    // crosses a 1024-token prefill boundary. Its host payload may be released after the first
    // encode, while later chunks must continue to reuse the resident Vision transient.
    ninfer::RequestOptions cross_chunk_options            = options(false);
    cross_chunk_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult cross_chunk =
        engine.generate(engine.prepare(first_input(gradient_ppm(1024, 1024))), cross_chunk_options);
    if (!cross_chunk.prompt.has_media || cross_chunk.generated_token_ids.size() != 1) {
        std::cerr << "cross-chunk Vision item did not complete after releasing its host payload\n";
        return 1;
    }

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_input(image_bytes)), options(true));
    if (!first.prompt.has_media || first.generated_token_ids.size() != 2 ||
        first.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "real Vision request did not complete through the public Engine\n";
        return 1;
    }

    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(followup_input(image_bytes, first)), options(true));
    if (reused.stats.reused_prompt_tokens == 0 || reused.stats.timings.vision_seconds != 0.0 ||
        reused.generated_token_ids.size() != 2) {
        std::cerr << "same-media continuation did not reuse the resident Vision prefix: reused="
                  << reused.stats.reused_prompt_tokens
                  << " vision=" << reused.stats.timings.vision_seconds << '\n';
        return 1;
    }

    std::vector<std::uint8_t> second_image = image_bytes;
    second_image.back() ^= 0x5aU;
    const ninfer::GenerationResult appended = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(true));
    if (appended.stats.reused_prompt_tokens == 0 ||
        !(appended.stats.timings.vision_seconds > 0.0) ||
        appended.generated_token_ids.size() != 2) {
        std::cerr << "new-media suffix did not preserve the old multimodal prefix: reused="
                  << appended.stats.reused_prompt_tokens
                  << " vision=" << appended.stats.timings.vision_seconds << '\n';
        return 1;
    }

    const ninfer::GenerationResult baseline = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(false));
    if (baseline.generated_token_ids.size() != 2 || baseline.stats.reused_prompt_tokens != 0 ||
        !(baseline.stats.timings.vision_seconds > 0.0)) {
        std::cerr << "uncached multimodal prefill did not recompute its media\n";
        return 1;
    }

    std::vector<std::uint8_t> changed_prefix = image_bytes;
    changed_prefix[changed_prefix.size() - 2] ^= 0x33U;
    const ninfer::GenerationResult miss = engine.generate(
        engine.prepare(appended_media_input(changed_prefix, first, reused, second_image)),
        options(true));
    if (miss.stats.reused_prompt_tokens != 0) {
        std::cerr << "changed media content incorrectly reused placeholder-token KV\n";
        return 1;
    }

    ninfer::RequestOptions mtp_options            = options(false);
    mtp_options.execution.requested_output_tokens = 5;
    mtp_options.execution.sampling.seed           = 0;
    const ninfer::GenerationResult mtp_baseline =
        engine.generate(engine.prepare(first_input(image_bytes)), mtp_options);
    if (mtp_baseline.generated_token_ids.size() != 5 ||
        mtp_baseline.generated_token_ids[0] == mtp_baseline.generated_token_ids[1]) {
        std::cerr << "multimodal stop fixture did not produce distinct leading tokens\n";
        return 1;
    }
    ninfer::RequestOptions stop_options = mtp_options;
    stop_options.stop.token_ids.push_back(mtp_baseline.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare(first_input(image_bytes)), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.empty() || stopped.stats.speculative.rounds == 0 ||
        stopped.generated_token_ids.back() != stop_options.stop.token_ids.front()) {
        std::cerr << "multimodal custom stop did not terminate at the selected token\n";
        return 1;
    }
    const ninfer::GenerationResult after_stop =
        engine.generate(engine.prepare(followup_input(image_bytes, stopped)), options(false));
    if (after_stop.generated_token_ids.size() != 2 ||
        after_stop.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "multimodal request after custom stop did not finish its output budget\n";
        return 1;
    }

    // Exact artifact rendering prefix before the first image-pad column:
    // <|im_start|>user\n<|vision_start|>. Reusing it places the MTP bridge directly on the first
    // Vision merger column rather than on an ordinary token embedding.
    const std::vector<ninfer::TokenId> visual_prefix{248045, 846, 198, 248053};
    ninfer::RequestOptions source_options            = options(true);
    source_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult bridge_source =
        engine.generate(engine.prepare_tokens(visual_prefix), source_options);
    ninfer::RequestOptions bridge_options            = options(true);
    bridge_options.execution.requested_output_tokens = 5;
    // Earlier checks retained complete image histories. Use new media so this request must
    // consume the short text-only source and actually exercise the visual MTP bridge.
    auto bridge_image = image_bytes;
    bridge_image.back() ^= 0x19U;
    const ninfer::GenerationResult visual_bridge =
        engine.generate(engine.prepare(first_input(bridge_image)), bridge_options);
    if (bridge_source.generated_token_ids.size() != 1 ||
        visual_bridge.stats.reused_prompt_tokens != visual_prefix.size() ||
        !(visual_bridge.stats.timings.vision_seconds > 0.0) ||
        visual_bridge.stats.speculative.rounds == 0) {
        std::cerr << "visual MTP bridge did not append the prefix and enter speculative decode: "
                  << "source_outputs=" << bridge_source.generated_token_ids.size()
                  << " reused=" << visual_bridge.stats.reused_prompt_tokens
                  << " vision=" << visual_bridge.stats.timings.vision_seconds
                  << " rounds=" << visual_bridge.stats.speculative.rounds
                  << " fallbacks=" << visual_bridge.stats.speculative.fallback_steps << '\n';
        return 1;
    }
    if (visual_bridge.generated_token_ids.size() != 5 ||
        visual_bridge.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "visual MTP bridge did not commit the requested output budget\n";
        return 1;
    }
    const auto bridge_followup =
        engine.generate(engine.prepare(followup_input(bridge_image, visual_bridge)), options(true));
    if (bridge_followup.stats.reused_prompt_tokens == 0 ||
        bridge_followup.stats.timings.vision_seconds != 0.0 ||
        bridge_followup.generated_token_ids.size() != 2) {
        std::cerr << "visual MTP bridge lost its retained continuation: reused="
                  << bridge_followup.stats.reused_prompt_tokens
                  << " vision=" << bridge_followup.stats.timings.vision_seconds
                  << " outputs=" << bridge_followup.generated_token_ids.size() << '\n';
        return 1;
    }
    return 0;
}

ninfer::PromptInput session_turn(std::string session, std::string question) {
    ninfer::PromptInput input;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    user.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(question), .media = {}});
    input.messages.push_back(std::move(user));
    input.options.enable_thinking   = false;
    input.context_cache.session_key = std::move(session);
    return input;
}

ninfer::RequestOptions fixed_output(std::uint32_t tokens, bool reuse = true) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = tokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = false;
    return options;
}

int exercise_concurrent_resource_settlement(const char* artifact) {
    ninfer::Engine engine(concurrent_engine_options(artifact));

    constexpr std::string_view kSession = "publication-order-real";
    constexpr std::string_view kOlderQuestion =
        "Describe deterministic scheduling using exactly one concise paragraph.";
    constexpr std::string_view kNewerQuestion =
        "Describe prefix caching using exactly one concise paragraph.";
    auto older = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kOlderQuestion))),
        fixed_output(24));
    auto newer = engine.submit(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    const ninfer::GenerationResult newer_result = newer.wait();
    const ninfer::GenerationResult older_result = older.wait();
    if (newer_result.generated_token_ids.size() != 2 ||
        older_result.generated_token_ids.size() != 24) {
        std::cerr << "concurrent session requests did not reach staggered terminal boundaries\n";
        return 1;
    }

    for (std::uint32_t index = 0; index < 6; ++index) {
        const std::string suffix              = std::to_string(index);
        const ninfer::GenerationResult filler = engine.generate(
            engine.prepare(session_turn("publication-filler-" + suffix,
                                        "Give one deterministic token for filler " + suffix + '.')),
            fixed_output(1));
        if (filler.generated_token_ids.size() != 1) {
            std::cerr << "session-order catalog filler did not complete\n";
            return 1;
        }
    }
    const ninfer::GenerationResult replay = engine.generate(
        engine.prepare(session_turn(std::string(kSession), std::string(kNewerQuestion))),
        fixed_output(2));
    if (replay.generated_token_ids.size() != 2 || replay.stats.reused_prompt_tokens == 0 ||
        replay.stats.prefix_reuse_path == ninfer::PrefixReusePath::Root) {
        std::cerr << "late older finish lost a reusable newer conversation: path="
                  << static_cast<int>(replay.stats.prefix_reuse_path)
                  << " reused=" << replay.stats.reused_prompt_tokens << '\n';
        return 1;
    }

    {
        std::vector<ninfer::TokenId> long_prompt(400, 198);
        auto cancelled =
            engine.submit(engine.prepare_tokens(std::move(long_prompt)), fixed_output(32, false));
        if (!cancelled) {
            std::cerr << "materialization cancellation fixture did not create a handle\n";
            return 1;
        }
    }
    const ninfer::GenerationResult after_cancel = engine.generate(
        engine.prepare_tokens({248045, 846, 198, 5834, 248046, 198}), fixed_output(1, false));
    if (after_cancel.generated_token_ids.size() != 1) {
        std::cerr << "request after materialization cancellation did not complete\n";
        return 1;
    }

    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(8);
    for (std::uint32_t row = 0; row < 8; ++row) {
        std::vector<ninfer::TokenId> prompt{
            248045, 846, 198, static_cast<ninfer::TokenId>(1000 + row), 248046, 198};
        handles.push_back(
            engine.submit(engine.prepare_tokens(std::move(prompt)), fixed_output(row + 1, false)));
    }
    for (std::uint32_t row = 0; row < handles.size(); ++row) {
        const ninfer::GenerationResult result = handles[row].wait();
        if (result.generated_token_ids.size() != row + 1 ||
            result.finish_reason != ninfer::FinishReason::OutputLimit) {
            std::cerr << "C=8 staggered row " << row << " did not terminate independently\n";
            return 1;
        }
    }
    const ninfer::RuntimeStats settled = engine.snapshot().runtime;
    if (settled.running_requests != 0 || settled.materializing_requests != 0 ||
        settled.paused_requests != 0 || settled.replaying_requests != 0 ||
        settled.prefilling_requests != 0 || settled.decode_ready_requests != 0 ||
        settled.capture_pending_requests != 0 || settled.terminal_pending_requests != 0) {
        std::cerr << "C=8 terminal settlement left live logical membership: running="
                  << settled.running_requests << " materializing=" << settled.materializing_requests
                  << " prefill=" << settled.prefilling_requests
                  << " decode=" << settled.decode_ready_requests
                  << " capture=" << settled.capture_pending_requests
                  << " terminal=" << settled.terminal_pending_requests << '\n';
        return 1;
    }
    return 0;
}

int verify_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.startup_report().load;
    if (load.architecture != "Qwen3_5ForCausalLM" || load.model_name.empty() ||
        load.weight_formats.empty() || load.host_to_device_bytes == 0 ||
        load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "Engine construction has an invalid load summary: target=" << load.architecture
                  << " weights=" << load.prefill_signature << '\n';
        return 1;
    }
    const auto memory  = engine.snapshot().ranks.at(0).memory;
    const auto* vision = memory.vision ? &*memory.vision : nullptr;
    if (memory.allocations.device_weights_bytes == 0 ||

        memory.allocations.persistent_device_bytes == 0 ||
        memory.allocations.workspace_device_bytes == 0 || vision == nullptr ||
        vision->max_window_features == 0 || vision->max_window_features > 4096 ||
        vision->encoder_workspace_bytes == 0 || vision->output_capacity_bytes == 0 ||
        vision->encoder_workspace_bytes + vision->output_capacity_bytes > vision->budget_bytes ||
        vision->output_capacity_bytes > memory.allocations.persistent_device_bytes ||
        vision->output_active_bytes != 0 ||
        !engine.startup_report().ranks.at(0).graphs.has_value()) {
        std::cerr << "Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

} // namespace

int exercise_artifact(const char* artifact) {
    {
        auto options = engine_options(artifact);
        ninfer::test::qwen3_5::set_context_capacity(options, 2);
        ninfer::Engine engine(std::move(options));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
        if (const int result = exercise_artifact_frontend(engine); result != 0) { return result; }
        if (const int result = exercise_stream_observations(engine); result != 0) { return result; }
        if (const int result = exercise_admission_wait(engine); result != 0) { return result; }
        if (const int result = exercise_full_prefill_chunk(engine); result != 0) { return result; }
        if (const int result = exercise_rewrite_checkpoints(engine); result != 0) { return result; }
        if (const int result = exercise_prefix(engine); result != 0) { return result; }
        if (const int result = exercise_abandoned_handle_capacity(engine); result != 0) {
            return result;
        }
    }
    if (const int result = exercise_rewrite_branch(artifact); result != 0) { return result; }
    if (const int result = exercise_late_instructions(artifact); result != 0) { return result; }
    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = exercise_vision(engine); result != 0) { return result; }
    }
    if (const int result = exercise_semantic_captures(artifact); result != 0) { return result; }
    if (const int result = exercise_host_restore(artifact); result != 0) { return result; }
    if (const int result = exercise_explicit_prefix(artifact); result != 0) { return result; }
    if (const int result = exercise_nested_tool_markers(artifact); result != 0) { return result; }
    if (const int result = exercise_explicit_anchor_branch(artifact); result != 0) {
        return result;
    }
    if (const int result = exercise_concurrent_resource_settlement(artifact); result != 0) {
        return result;
    }
    return 0;
}

int exercise_attention_integration(const char* artifact) {
    const auto setting = [](const char* name, const char* fallback) {
        const char* value = std::getenv(name);
        return std::string_view(value && *value ? value : fallback);
    };
    const auto storage =
        ninfer::test::parse_kv_cache_storage(setting("NINFER_TEST_KV_DTYPE", "bf16"));
    const auto backend_name = setting("NINFER_TEST_SPECULATIVE", "mtp");
    ninfer::SpeculativeBackend backend;
    if (backend_name == "none")
        backend = ninfer::SpeculativeBackend::None;
    else if (backend_name == "mtp")
        backend = ninfer::SpeculativeBackend::Mtp;
    else if (backend_name == "dflash")
        backend = ninfer::SpeculativeBackend::DFlash;
    else if (backend_name == "dflash2")
        backend = ninfer::SpeculativeBackend::DFlash2;
    else
        throw std::invalid_argument("unknown attention integration backend");
    const auto batch =
        static_cast<std::uint32_t>(std::stoul(std::string(setting("NINFER_TEST_BATCH", "2"))));
    const auto drafts = static_cast<std::uint32_t>(std::stoul(std::string(setting(
        "NINFER_TEST_DRAFT_TOKENS", backend == ninfer::SpeculativeBackend::Mtp ? "3" : "7"))));
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.kv_cache                  = storage;
    options.max_context               = 65536;
    options.prefill_chunk             = 1024;
    options.max_concurrency           = batch;
    options.max_pending_requests      = batch;
    options.speculative.backend       = backend;
    options.speculative.draft_tokens  = backend == ninfer::SpeculativeBackend::None ? 0 : drafts;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    ninfer::test::qwen3_5::set_context_capacity(options, batch + 2);
    ninfer::Engine engine(options);
    if (engine.description().kv_cache != storage)
        throw std::runtime_error("attention integration selected the wrong KV dtype");

    const auto validate = [backend](const ninfer::GenerationResult& result, std::uint32_t count) {
        if (result.generated_token_ids.size() != count ||
            result.finish_reason != ninfer::FinishReason::OutputLimit)
            throw std::runtime_error("attention integration did not finish its output budget");
        if (count > 1 && backend != ninfer::SpeculativeBackend::None &&
            result.stats.speculative.rounds == 0)
            throw std::runtime_error(
                "attention integration did not exercise speculative verification");
    };
    const auto seed =
        engine.tokenize_text("Explain how a sequence continues from its stored prefix. ");
    const auto visible_offset = backend == ninfer::SpeculativeBackend::Mtp ? 2 * drafts : 0;
    std::vector<ninfer::TokenId> prefix(8182 - visible_offset - 3 * (batch - 1));
    for (std::size_t i = 0; i < prefix.size(); ++i) prefix[i] = seed[i % seed.size()];
    const auto primed = engine.generate(engine.prepare_tokens(prefix), fixed_output(8));
    validate(primed, 8);
    prefix.insert(prefix.end(), primed.generated_token_ids.begin(),
                  primed.generated_token_ids.end());

    // Long output budgets keep rows active together even when another row prefills, across a
    // Graph resource tier.
    const auto before = engine.snapshot().runtime;
    std::vector<ninfer::GenerationHandle> handles;
    for (std::uint32_t row = 0; row < batch; ++row) {
        auto prompt = prefix;
        prompt.insert(prompt.end(), row * 3, seed.back());
        handles.push_back(
            engine.submit(engine.prepare_tokens(prompt), fixed_output(128 + row * 5)));
    }
    std::vector<ninfer::TokenId> continuation = prefix;
    std::uint64_t reused_tokens               = 0;
    for (std::uint32_t row = 0; row < batch; ++row) {
        const auto result = handles[row].wait();
        validate(result, 128 + row * 5);
        reused_tokens += result.stats.reused_prompt_tokens;
        if (row == 0)
            continuation.insert(continuation.end(), result.generated_token_ids.begin(),
                                result.generated_token_ids.end());
    }
    if (reused_tokens == 0)
        throw std::runtime_error("attention integration did not reuse the primed continuation");
    const auto after = engine.snapshot().runtime;
    if (batch > 1 && after.decode_row_rounds - before.decode_row_rounds <=
                         after.decode_rounds - before.decode_rounds)
        throw std::runtime_error("attention integration did not execute a multi-row decode round");
    continuation.insert(continuation.end(), 33, seed.back());
    const auto reused = engine.generate(engine.prepare_tokens(continuation), fixed_output(8));
    const auto fresh = engine.generate(engine.prepare_tokens(continuation), fixed_output(8, false));
    validate(reused, 8);
    validate(fresh, 8);
    if (reused.stats.reused_prompt_tokens == 0 || fresh.stats.reused_prompt_tokens != 0)
        throw std::runtime_error(
            "attention integration prefix continuation did not follow reuse policy");
    const auto appended = reused.prompt.prompt_tokens - reused.stats.reused_prompt_tokens;
    if (appended <= 16 || appended > 256)
        throw std::runtime_error("attention integration did not exercise small prefix append");
    const auto memory = engine.snapshot().ranks.at(0).memory;
    if (memory.workspace.measurement_peak_bytes == 0 ||
        memory.workspace.measurement_peak_bytes > memory.allocations.workspace_device_bytes)
        throw std::runtime_error("attention integration exceeded its planned workspace");
    std::cout << "attention integration KV=" << setting("NINFER_TEST_KV_DTYPE", "bf16")
              << " backend=" << backend_name << " B=" << batch
              << " workspace_peak=" << memory.workspace.measurement_peak_bytes
              << " graph_rebuild_estimate="
              << engine.startup_report()
                     .ranks.at(0)
                     .graphs.value_or(ninfer::GraphMemorySummary{})
                     .rebuild_estimate_bytes
              << " appended=" << appended
              << " append_prefill_ms=" << reused.stats.timings.prefill_seconds * 1000 << '\n';
    return 0;
}

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const char* selected            = std::getenv("NINFER_PREFIX_REAL_SCENARIO");
    const std::string_view scenario = selected ? selected : "all";
    int result                      = 0;
    if (scenario == "attention") {
        try {
            result = exercise_attention_integration(artifact);
        } catch (const std::exception& error) {
            std::cerr << "attention integration failed: " << error.what() << '\n';
            return 1;
        }
    } else if (scenario == "vision") {
        ninfer::Engine engine(engine_options(artifact));
        result = exercise_vision(engine);
    } else if (scenario == "all") {
        result = exercise_artifact(artifact);
    } else if (scenario == "concurrent") {
        result = exercise_concurrent_resource_settlement(artifact);
    } else if (scenario == "anthropic-prefix-regression") {
        result = exercise_anthropic_prefix_regression(artifact);
    } else if (scenario == "nested-tool-markers") {
        result = exercise_nested_tool_markers(artifact);
    } else if (scenario == "shared-rewrite-materialization") {
        result = exercise_shared_rewrite_materialization(artifact);
    } else if (scenario == "late-instructions") {
        result = exercise_late_instructions(artifact);
    } else if (scenario == "agent-continuation") {
        result = exercise_agent_continuation(artifact);
    } else if (scenario == "explicit-prefix") {
        result = exercise_explicit_prefix(artifact);
    } else if (scenario == "explicit-anchor") {
        result = exercise_explicit_anchor_branch(artifact);
    } else if (scenario == "rewrite-checkpoint") {
        auto options = engine_options(artifact);
        ninfer::test::qwen3_5::set_context_capacity(options, 2);
        ninfer::Engine engine(std::move(options));
        result = exercise_rewrite_checkpoints(engine);
    } else if (scenario == "terminal-publication" || scenario == "stream-observations") {
        auto options          = engine_options(artifact);
        options.enable_vision = scenario == "terminal-publication";
        if (options.enable_vision) { options.vision_memory_budget_bytes = 64ULL << 20; }
        options.context_cache = ninfer::ContextCacheOptions{.enabled = false};
        ninfer::test::qwen3_5::set_context_capacity(options);
        ninfer::Engine engine(std::move(options));
        if (scenario == "terminal-publication") {
            result = exercise_abandoned_handle_capacity(engine);
            if (result == 0) { result = exercise_terminal_publication(engine); }
        } else {
            result = exercise_stream_observations(engine);
            if (result == 0) { result = exercise_admission_wait(engine); }
            if (result == 0) { result = exercise_abandoned_handle_capacity(engine); }
        }
    } else {
        throw std::invalid_argument("unknown prefix integration scenario");
    }
    if (result == 0) { std::cout << "ok\n"; }
    return result;
}
