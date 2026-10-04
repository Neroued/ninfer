#include "ninfer/engine.h"
#include "kv_cache_storage.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                       = artifact;
    options.max_context                         = 4096;
    options.kv_capacity                         = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                       = 1024;
    options.speculative.backend                 = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens            = 3;
    options.speculative.proposal_head           = ninfer::ProposalHead::Optimized;
    options.enable_vision                       = true;
    options.max_concurrency                     = 1;
    options.max_pending_requests                = 1;
    options.context_cache.device_snapshot_slots = 4;
    return options;
}

ninfer::EngineOptions concurrent_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                       = artifact;
    options.max_context                         = 512;
    options.kv_capacity                         = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk                       = 256;
    options.speculative.backend                 = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens            = 3;
    options.speculative.proposal_head           = ninfer::ProposalHead::Optimized;
    options.max_concurrency                     = 8;
    options.max_pending_requests                = 8;
    options.context_cache.host_cache_bytes      = 0;
    options.context_cache.device_snapshot_slots = 16;
    return options;
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

int exercise_registered_frontend(const ninfer::Engine& engine) {
    if (engine.count_tokens(chinese_chat(true)) != 16) {
        std::cerr << "registered tokenizer/chat template changed the thinking prompt golden\n";
        return 1;
    }
    if (engine.count_tokens(chinese_chat(false)) != 18) {
        std::cerr << "registered tokenizer/chat template changed the no-thinking prompt golden\n";
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

    void publish(ninfer::OutputDelta) override {
        if (!timing_seen_) { valid_ = false; }
    }

    [[nodiscard]] bool valid_for(const ninfer::GenerationResult& result) const {
        return valid_ && started_ && timing_seen_ && start_.reused_prompt_tokens == 0 &&
               last_processed_ == start_.prompt.prompt_tokens &&
               last_timing_.generated_tokens == result.generated_token_ids.size() &&
               result.timings.prompt_wall_seconds > 0.0 &&
               result.timings.generation_wall_seconds >= 0.0;
    }

private:
    ninfer::GenerationStart start_;
    ninfer::GenerationTimingObservation last_timing_;
    std::uint32_t last_processed_           = 0;
    std::uint64_t last_progress_elapsed_ns_ = 0;
    bool started_                           = false;
    bool timing_seen_                       = false;
    bool valid_                             = true;
};

int exercise_stream_observations(ninfer::Engine& engine) {
    std::vector<ninfer::TokenId> prompt(2050, 198);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = 3;
    request.execution.sampling.temperature    = 0.0F;
    request.execution.allow_prefix_reuse      = false;
    request.stop.include_model_defaults       = false;
    const ninfer::GenerationObservationOptions observation{
        .phase_timings = true, .live_timings = true, .prompt_progress = true};

    ObservationSink sink;
    ninfer::GenerationHandle generation =
        engine.submit(engine.prepare_tokens(std::move(prompt)), std::move(request),
                      ninfer::OutputConsumerMode::Streaming, observation);
    const ninfer::GenerationResult result = generation.wait(&sink);
    if (result.generated_token_ids.size() != 3 || !sink.valid_for(result)) {
        std::cerr
            << "stream observations lost prompt progress, commit timing, or publication order\n";
        return 1;
    }
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
    const ninfer::RuntimeStats settled = engine.runtime_stats();
    if (settled.running_requests != 0 || settled.materializing_requests != 0 ||
        settled.prefilling_requests != 0 || settled.decode_ready_requests != 0 ||
        settled.terminal_pending_requests != 0) {
        std::cerr << "C=8 terminal settlement left live logical membership: running="
                  << settled.running_requests << " materializing=" << settled.materializing_requests
                  << " prefill=" << settled.prefilling_requests
                  << " decode=" << settled.decode_ready_requests
                  << " terminal=" << settled.terminal_pending_requests << '\n';
        return 1;
    }
    return 0;
}

int verify_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.architecture != "Qwen3_5ForCausalLM" || load.model_name.empty() ||
        load.weight_formats.empty() || load.host_to_device_bytes == 0 ||
        load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "Engine construction has an invalid load summary: target=" << load.architecture
                  << " weights=" << load.prefill_signature << '\n';
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    const auto* vision = memory.vision_workspace ? &*memory.vision_workspace : nullptr;
    if (memory.weights.capacity_bytes == 0 || memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || vision == nullptr ||
        vision->aggregate_prompt_tokens != 4096 || vision->max_item_tokens != 4096 ||
        vision->general_capacity_bytes == 0 || vision->encode_peak_bytes == 0 ||
        vision->handoff_offset_bytes > memory.workspace.capacity_bytes ||
        vision->handoff_capacity_bytes == 0 ||
        vision->handoff_capacity_bytes >
            memory.workspace.capacity_bytes - vision->handoff_offset_bytes ||
        vision->handoff_active_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

} // namespace

int exercise_artifact(const char* artifact) {
    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
        if (const int result = exercise_registered_frontend(engine); result != 0) { return result; }
        if (const int result = exercise_stream_observations(engine); result != 0) { return result; }
        if (const int result = exercise_full_prefill_chunk(engine); result != 0) { return result; }
        if (const int result = exercise_abandoned_handle_capacity(engine); result != 0) {
            return result;
        }
    }
    return exercise_concurrent_resource_settlement(artifact);
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
    options.artifact_path                    = artifact;
    options.kv_cache                         = storage;
    options.max_context                      = 65536;
    options.kv_capacity                      = ninfer::KvCapacityPolicy::explicit_capacity(65536);
    options.prefill_chunk                    = 1024;
    options.max_concurrency                  = batch;
    options.max_pending_requests             = batch;
    options.context_cache.device_snapshot_slots = batch + 2;
    options.speculative.backend                     = backend;
    options.speculative.draft_tokens  = backend == ninfer::SpeculativeBackend::None ? 0 : drafts;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    ninfer::Engine engine(options);
    if (engine.memory_summary().kv_cache != storage)
        throw std::runtime_error("attention integration selected the wrong KV dtype");

    const auto validate = [backend](const ninfer::GenerationResult& result, std::uint32_t count) {
        if (result.generated_token_ids.size() != count ||
            result.finish_reason != ninfer::FinishReason::OutputLimit)
            throw std::runtime_error("attention integration did not finish its output budget");
        if (count > 1 && backend != ninfer::SpeculativeBackend::None &&
            result.speculative.rounds == 0)
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
    const auto before = engine.runtime_stats();
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
        reused_tokens += result.reused_prompt_tokens;
        if (row == 0)
            continuation.insert(continuation.end(), result.generated_token_ids.begin(),
                                result.generated_token_ids.end());
    }
    if (reused_tokens == 0)
        throw std::runtime_error("attention integration did not reuse the primed continuation");
    const auto after = engine.runtime_stats();
    if (batch > 1 && after.decode_row_rounds - before.decode_row_rounds <=
                         after.decode_rounds - before.decode_rounds)
        throw std::runtime_error("attention integration did not execute a multi-row decode round");
    continuation.insert(continuation.end(), 33, seed.back());
    const auto reused = engine.generate(engine.prepare_tokens(continuation), fixed_output(8));
    const auto fresh = engine.generate(engine.prepare_tokens(continuation), fixed_output(8, false));
    validate(reused, 8);
    validate(fresh, 8);
    if (reused.reused_prompt_tokens == 0 || fresh.reused_prompt_tokens != 0)
        throw std::runtime_error(
            "attention integration prefix continuation did not follow reuse policy");
    const auto appended = reused.prompt.prompt_tokens - reused.reused_prompt_tokens;
    if (appended <= 16 || appended > 256)
        throw std::runtime_error("attention integration did not exercise small prefix append");
    const auto memory = engine.memory_summary();
    if (memory.workspace_logical_peak_bytes == 0 ||
        memory.workspace_logical_peak_bytes > memory.workspace.capacity_bytes)
        throw std::runtime_error("attention integration exceeded its planned workspace");
    std::cout << "attention integration KV=" << setting("NINFER_TEST_KV_DTYPE", "bf16")
              << " backend=" << backend_name << " B=" << batch
              << " workspace_peak=" << memory.workspace_logical_peak_bytes
              << " graph_allowance=" << memory.cuda_graph_allowance_bytes
              << " appended=" << appended
              << " append_prefill_ms=" << reused.timings.prefill_seconds * 1000 << '\n';
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
    } else if (scenario == "all") {
        result = exercise_artifact(artifact);
    } else if (scenario == "concurrent") {
        result = exercise_concurrent_resource_settlement(artifact);
    } else if (scenario == "stream-observations") {
        auto options          = engine_options(artifact);
        options.enable_vision = false;
        options.context_cache = ninfer::ContextCacheOptions{.enabled = false};
        ninfer::Engine engine(std::move(options));
        result = exercise_stream_observations(engine);
    } else {
        throw std::invalid_argument("unknown prefix integration scenario");
    }
    if (result == 0) { std::cout << "ok\n"; }
    return result;
}
