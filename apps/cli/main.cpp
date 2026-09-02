#include "options.h"
#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "product/prompt_input/prompt_input.h"

#include "ninfer/engine.h"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#include <spdlog/logger.h>

namespace {

std::string format_seconds(double seconds) {
    std::ostringstream output;
    output << std::fixed << std::setprecision(3) << seconds << " s";
    return output.str();
}

std::string format_rate(double tokens, double seconds) {
    if (tokens <= 0.0 || seconds <= 0.0) { return "n/a"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << tokens / seconds << " tok/s";
    return output.str();
}

std::string format_percent(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) { return "n/a"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2)
           << 100.0 * static_cast<double>(numerator) / static_cast<double>(denominator) << '%';
    return output.str();
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr double kKiB = 1024.0;
    constexpr double kMiB = 1024.0 * kKiB;
    constexpr double kGiB = 1024.0 * kMiB;
    std::ostringstream output;
    output << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        output << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        output << static_cast<double>(bytes) / kMiB << " MiB";
    } else if (bytes >= static_cast<std::uint64_t>(kKiB)) {
        output << static_cast<double>(bytes) / kKiB << " KiB";
    } else {
        output << bytes << " B";
    }
    return output.str();
}

std::string format_arena_used(const ninfer::ArenaMemorySummary& arena) {
    return format_bytes(arena.used_bytes) + " / " + format_bytes(arena.capacity_bytes);
}

std::string format_arena_peak(const ninfer::ArenaMemorySummary& arena) {
    return format_bytes(arena.peak_used_bytes) + " / " + format_bytes(arena.capacity_bytes);
}

std::string format_sampling(const ninfer::ResolvedSamplingParameters& sampling) {
    if (sampling.temperature <= 0.0F) { return "greedy (temperature 0)"; }
    std::ostringstream output;
    output << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
           << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k
           << " min_p=" << sampling.min_p << " presence=" << sampling.presence_penalty
           << " freq=" << sampling.frequency_penalty << " seed=" << sampling.seed;
    return output.str();
}

std::string format_finish(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output-limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context-capacity";
    case ninfer::FinishReason::StopToken:
        return "stop-token";
    case ninfer::FinishReason::StopString:
        return "stop-string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

std::string format_kv_cache(ninfer::KvCacheStorage storage) {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8-group64";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-row256";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

std::string format_kv_capacity_mode(ninfer::KvCapacityMode mode) {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

void print_stage(std::string_view group, std::string_view detail, double seconds) {
    std::cerr << std::left << std::setw(12) << group << std::setw(26) << detail << std::right
              << std::setw(12) << format_seconds(seconds) << '\n';
}

void print_metric(std::string_view label, std::string_view value) {
    std::cerr << std::left << std::setw(12) << "summary" << std::setw(26) << label << value << '\n';
}

class StreamingSink final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart) override {}

    void publish(ninfer::OutputDelta delta) override {
        std::ostream& output =
            delta.channel == ninfer::OutputChannel::Reasoning ? std::cerr : std::cout;
        output << delta.text;
        output.flush();
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            reasoning_seen_ = reasoning_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { reasoning_ends_in_newline_ = delta.text.back() == '\n'; }
        } else {
            content_seen_ = content_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { content_ends_in_newline_ = delta.text.back() == '\n'; }
        }
    }

    void finish_streams(bool successful = true) {
        if (finished_) { return; }
        finished_ = true;
        if ((successful && !content_seen_) || (content_seen_ && !content_ends_in_newline_)) {
            std::cout << '\n';
        }
        std::cout.flush();
        if (reasoning_seen_ && !reasoning_ends_in_newline_) { std::cerr << '\n'; }
    }

private:
    bool content_seen_              = false;
    bool content_ends_in_newline_   = false;
    bool reasoning_seen_            = false;
    bool reasoning_ends_in_newline_ = false;
    bool finished_                  = false;
};

void print_generation_summary(const ninfer::GenerationResult& result,
                              const ninfer::ResolvedSamplingParameters& sampling,
                              const ninfer::MemorySummary& memory) {
    print_stage("prepare", "render/preprocess", result.timings.prepare_seconds);
    print_stage("generate", "vision", result.timings.vision_seconds);
    print_stage("generate", "text prefill", result.timings.prefill_seconds);
    print_stage("generate", "decode", result.timings.decode_seconds);
    print_stage("generate", "total", result.timings.total_seconds);

    const std::size_t generated = result.generated_token_ids.size();
    const std::size_t decoded   = generated == 0 ? 0 : generated - 1;
    const double model_seconds  = result.timings.vision_seconds + result.timings.prefill_seconds +
                                 result.timings.decode_seconds;
    print_metric("sampling", format_sampling(sampling));
    print_metric("finish reason", format_finish(result.finish_reason));
    print_metric("prompt tokens", std::to_string(result.prompt.prompt_tokens));
    print_metric("reused prompt tokens", std::to_string(result.reused_prompt_tokens));
    print_metric("generated tokens", std::to_string(generated));
    if (result.thinking.configured_budget) {
        print_metric("thinking budget", std::to_string(*result.thinking.configured_budget));
        print_metric("model thinking tokens",
                     std::to_string(result.thinking.model_thinking_tokens));
        print_metric("thinking control tokens", std::to_string(result.thinking.injected_tokens));
        print_metric("thinking control", result.thinking.applied ? "applied" : "not applied");
    }
    print_metric("model elapsed", format_seconds(model_seconds));
    print_metric("prefill speed", format_rate(static_cast<double>(result.prompt.prompt_tokens),
                                              result.timings.prefill_seconds));
    print_metric("decode speed",
                 format_rate(static_cast<double>(decoded), result.timings.decode_seconds));
    print_metric("throughput (overall)",
                 format_rate(static_cast<double>(generated), model_seconds));

    const std::uint64_t reserved = static_cast<std::uint64_t>(memory.weights.capacity_bytes) +
                                   memory.runtime_reservation_bytes;
    print_metric("device", std::to_string(memory.device));
    print_metric("max context", std::to_string(memory.max_context));
    print_metric("KV capacity policy", format_kv_capacity_mode(memory.kv_capacity_mode));
    print_metric("KV capacity", std::to_string(memory.kv_capacity));
    print_metric("KV page groups", std::to_string(memory.kv_capacity_page_groups) + " / " +
                                       std::to_string(memory.kv_capacity_max_page_groups));
    print_metric("gpu weights used", format_arena_used(memory.weights));
    print_metric("gpu sequence used", format_arena_used(memory.sequence));
    print_metric("kv cache dtype", format_kv_cache(memory.kv_cache));
    print_metric("kv cache payload", format_bytes(memory.kv_payload_bytes));
    print_metric("gpu workspace peak", format_arena_peak(memory.workspace));
    print_metric("runtime reservation", format_bytes(memory.runtime_reservation_bytes));
    print_metric("free after weights", format_bytes(memory.available_after_weights_bytes));
    print_metric("free after startup", format_bytes(memory.available_after_startup_bytes));
    print_metric("KV capacity headroom", format_bytes(memory.kv_capacity_headroom_bytes));
    print_metric("planned slack", format_bytes(memory.planned_slack_bytes));
    print_metric("CUDA Graph allowance", format_bytes(memory.cuda_graph_allowance_bytes));
    print_metric("planned device total", format_bytes(reserved));

    const ninfer::SpeculativeStats& speculative = result.speculative;
    if (speculative.enabled) {
        const std::string backend =
            speculative.backend == ninfer::SpeculativeBackend::DFlash ? "dflash" : "mtp";
        print_metric(backend + " draft window", std::to_string(speculative.draft_window));
        print_metric(backend + " rounds", std::to_string(speculative.rounds));
        print_metric(backend + " fallback steps", std::to_string(speculative.fallback_steps));
        print_metric(backend + " drafted tokens", std::to_string(speculative.drafted_tokens));
        print_metric(backend + " accepted tokens", std::to_string(speculative.accepted_tokens));
        print_metric(backend + " acceptance rate",
                     format_percent(speculative.accepted_tokens, speculative.drafted_tokens));
        if (speculative.rounds != 0) {
            std::ostringstream length;
            length << std::fixed << std::setprecision(2)
                   << 1.0 + static_cast<double>(speculative.accepted_tokens) /
                                static_cast<double>(speculative.rounds)
                   << " tok/round";
            print_metric(backend + " acceptance length", length.str());
        }
        if (!speculative.accepted_per_position.empty()) {
            std::ostringstream positions;
            for (std::size_t i = 0; i < speculative.accepted_per_position.size(); ++i) {
                if (i != 0) { positions << ','; }
                positions << speculative.accepted_per_position[i];
            }
            print_metric(backend + " accepted by pos", positions.str());
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    ninfer::cli::Options cli;
    try {
        cli = ninfer::cli::parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << ninfer::cli::usage_text(argv[0]);
        return 1;
    }
    if (cli.help_requested) {
        std::cout << ninfer::cli::usage_text(argv[0]);
        return 0;
    }

    ninfer::product::LoggingRuntime logging({.logger_name = "ninfer"});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);

    try {

        ninfer::PromptInput input =
            cli.messages_path.empty()
                ? ninfer::product::prompt_from_text(cli.prompt, cli.enable_thinking)
                : ninfer::product::prompt_from_messages(cli.messages_path, cli.enable_thinking,
                                                        cli.enable_vision);
        input.options.reasoning_effort = cli.reasoning_effort;

        ninfer::RequestOptions request;
        request.execution.sampling                = cli.sampling;
        request.execution.requested_output_tokens = cli.max_new;
        request.execution.thinking.budget         = cli.thinking_budget;
        request.stop.token_ids                    = cli.stop_token_ids;
        request.stop.strings                      = cli.stop_strings;
        request.output.raw                        = cli.raw_output;

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path  = cli.artifact_path;
        engine_options.device         = cli.device;
        engine_options.max_context    = cli.max_context;
        engine_options.kv_capacity    = cli.kv_capacity;
        engine_options.prefill_chunk  = cli.prefill_chunk;
        engine_options.kv_cache       = cli.kv_cache;
        engine_options.speculative    = cli.speculative;
        engine_options.enable_vision  = cli.enable_vision;
        engine_options.use_cuda_graph = cli.use_cuda_graph;
        // One CLI invocation owns exactly one request, so retained cross-request context has no
        // consumer and must not reserve an extra Device StateImage or run terminal capture.
        engine_options.context_cache.enabled                = false;
        engine_options.context_cache.host_state_slots       = 0;
        engine_options.context_cache.host_kv_capacity_bytes = 0;
        engine_options.startup_observer                     = startup_log.observer();

        ninfer::Engine engine(std::move(engine_options));
        startup_log.engine_ready(engine.load_summary());
        engine.reset_memory_peaks();

        ninfer::PreparedPrompt prompt = engine.prepare(std::move(input));

        StreamingSink sink;
        ninfer::GenerationHandle generation = engine.submit(std::move(prompt), std::move(request),
                                                            ninfer::OutputConsumerMode::Streaming);
        const ninfer::ResolvedSamplingParameters sampling = generation.resolved_sampling();
        ninfer::GenerationResult result;
        try {
            result = generation.wait(&sink);
            sink.finish_streams();
        } catch (...) {
            sink.finish_streams(false);
            throw;
        }

        std::cerr << "phase       detail                      elapsed/progress\n";
        if (cli.print_token_ids) {
            std::cerr << std::left << std::setw(12) << "tokens" << std::setw(26) << "generated ids";
            for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << result.generated_token_ids[i];
            }
            std::cerr << '\n';
        }
        print_generation_summary(result, sampling, engine.memory_summary());
        return 0;
    } catch (const std::exception& error) {
        logger->error("application status=failed detail={}",
                      ninfer::product::quote_log_value(error.what()));
        return 1;
    }
}
