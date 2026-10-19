#include "product/config/options.h"
#include "product/logging/logging.h"
#include "product/logging/configuration.h"
#include "product/logging/pretty_format.h"
#include "product/logging/startup_log.h"
#include "product/prompt_input/prompt_input.h"
#include "product/speculative_options.h"

#include "ninfer/engine.h"
#include "product/statistics.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>

#include <spdlog/logger.h>

namespace {

std::string format_seconds(double seconds) {
    return ninfer::product::format_pretty_duration(seconds);
}

std::string format_rate(double tokens, double seconds) {
    if (tokens <= 0.0 || seconds <= 0.0) { return "n/a"; }
    return ninfer::product::format_pretty_rate(tokens / seconds, "tok");
}

std::string format_percent(std::uint64_t numerator, std::uint64_t denominator) {
    if (denominator == 0) { return "n/a"; }
    return ninfer::product::format_pretty_percent(static_cast<double>(numerator) /
                                                  static_cast<double>(denominator));
}

std::string format_bytes(std::uint64_t bytes) {
    return ninfer::product::format_pretty_bytes(bytes);
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

    void progress(ninfer::PromptProgress) override {}

    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputEvent event) override {
        if (const auto* call = std::get_if<ninfer::ToolCallStart>(&event)) {
            if (content_seen_ && !content_ends_in_newline_) std::cout << '\n';
            std::cout << "{\"name\":\"" << call->name << "\",\"arguments\":";
            content_seen_            = true;
            content_ends_in_newline_ = false;
        } else if (const auto* args = std::get_if<ninfer::ToolArgumentsDelta>(&event)) {
            std::cout << args->text;
        } else if (const auto* end = std::get_if<ninfer::ToolCallEnd>(&event)) {
            if (end->status == ninfer::ToolCallStatus::Complete)
                std::cout << '}';
            else
                std::cerr << "\nTool call interrupted; arguments are incomplete.\n";
            std::cout << '\n';
            content_ends_in_newline_ = true;
        } else {
            publish_text(std::get<ninfer::OutputDelta>(std::move(event)));
        }
        std::cout.flush();
    }

    void publish_text(ninfer::OutputDelta delta) {
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
                              const ninfer::Engine& engine) {
    print_stage("prepare", "request preparation", result.stats.timings.prepare_seconds);
    print_stage("gpu interval", "vision stream", result.stats.timings.vision_seconds);
    print_stage("generate", "prefill elapsed", result.stats.timings.prefill_seconds);
    print_stage("generate", "decode", result.stats.timings.decode_seconds);
    print_stage("generate", "total", result.stats.timings.total_seconds);

    const std::size_t generated = result.generated_token_ids.size();
    const std::size_t decoded   = generated == 0 ? 0 : generated - 1;
    const double model_seconds =
        result.stats.timings.prefill_seconds + result.stats.timings.replay_seconds +
        result.stats.timings.control_seconds + result.stats.timings.decode_seconds;
    print_metric("sampling", format_sampling(sampling));
    print_metric("finish reason", format_finish(result.finish_reason));
    print_metric("prompt tokens", std::to_string(result.prompt.prompt_tokens));
    print_metric("reused prompt tokens", std::to_string(result.stats.reused_prompt_tokens));
    print_metric("generated tokens", std::to_string(generated));
    if (result.stats.thinking.configured_budget) {
        print_metric("thinking budget", std::to_string(*result.stats.thinking.configured_budget));
        print_metric("model thinking tokens",
                     std::to_string(result.stats.thinking.model_thinking_tokens));
        print_metric("thinking control tokens",
                     std::to_string(result.stats.thinking.injected_tokens));
        print_metric("thinking control", result.stats.thinking.applied ? "applied" : "not applied");
    }
    print_metric("model elapsed", format_seconds(model_seconds));
    print_metric("prefill speed",
                 format_rate(static_cast<double>(result.stats.computed_prefill_tokens),
                             result.stats.timings.prefill_seconds));
    print_metric("decode speed", format_rate(static_cast<double>(decoded),
                                             result.stats.timings.generation_wall_seconds));
    print_metric("throughput (overall)",
                 format_rate(static_cast<double>(generated), model_seconds));

    const auto& description = engine.description();
    const auto& startup     = engine.startup_report();
    const auto snapshot     = engine.snapshot();
    print_metric("max context", std::to_string(description.max_context));
    print_metric("prefill chunk", std::to_string(description.prefill_chunk));
    print_metric("KV capacity", std::to_string(description.kv_capacity));
    print_metric("kv cache dtype", format_kv_cache(description.kv_cache));
    for (std::size_t i = 0; i < description.ranks.size(); ++i) {
        const auto& rank   = description.ranks[i];
        const auto& budget = startup.ranks.at(i);
        const auto found   = std::find_if(snapshot.ranks.begin(), snapshot.ranks.end(),
                                          [&](const auto& value) { return value.rank == rank.rank; });
        if (found == snapshot.ranks.end()) continue;
        const auto& memory       = found->memory;
        const std::string prefix = "rank " + std::to_string(rank.rank) + " ";
        print_metric(prefix + "device budget", format_bytes(budget.limit_bytes));
        print_metric(prefix + "GPU weights", format_bytes(memory.allocations.device_weights_bytes));
        print_metric(prefix + "GPU persistent",
                     format_bytes(memory.allocations.persistent_device_bytes));
        print_metric(prefix + "GPU workspace",
                     format_bytes(memory.allocations.workspace_device_bytes));
        print_metric(prefix + "workspace peak",
                     format_bytes(memory.workspace.measurement_peak_bytes));
        print_metric(prefix + "GPU allocations", format_bytes(memory.allocations.device_bytes()));
        print_metric(prefix + "Host allocations", format_bytes(memory.allocations.host_bytes()));
        print_metric(prefix + "KV payload", format_bytes(memory.kv_payload_bytes));
        print_metric(prefix + "startup free", format_bytes(budget.available_after_startup_bytes));
        if (memory.experts) {
            print_metric(prefix + "expert cache", format_bytes(memory.experts->cache_bytes));
            print_metric(prefix + "expert ready", format_bytes(memory.experts->ready_bytes));
        }
    }

    const ninfer::SpeculativeStats& speculative = result.stats.speculative;
    if (speculative.enabled) {
        const std::string backend = ninfer::product::speculative_backend_name(speculative.backend);
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
    ninfer::product::CliOptions cli;
    try {
        cli = ninfer::product::parse_cli_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        std::cerr << ninfer::product::startup_usage_text(argv[0], ninfer::product::EntryPoint::Cli);
        return 1;
    }
    if (cli.help_requested) {
        std::cout << ninfer::product::startup_usage_text(argv[0], ninfer::product::EntryPoint::Cli);
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer",
         .level        = cli.launch.logging.level,
         .presentation = ninfer::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);

    try {
        ninfer::product::log_configuration(*logger, ninfer::product::configuration_view(cli));
        const auto& defaults = cli.launch.generation;
        ninfer::PromptInput input =
            cli.messages_path.empty()
                ? ninfer::product::prompt_from_text(cli.prompt, defaults.enable_thinking)
                : ninfer::product::prompt_from_messages(cli.messages_path, defaults.enable_thinking,
                                                        cli.launch.engine.enable_vision);
        input.options.reasoning_effort  = cli.reasoning_effort;
        input.options.preserve_thinking = defaults.preserve_thinking;

        ninfer::RequestOptions request;
        if (cli.constraint.kind == ninfer::product::ConstraintKind::JsonObject)
            request.constraint = ninfer::OutputConstraint::json_object();
        else if (cli.constraint.kind == ninfer::product::ConstraintKind::Regex)
            request.constraint = ninfer::OutputConstraint::regex(cli.constraint.regex);
        else if (cli.constraint.kind == ninfer::product::ConstraintKind::Choice)
            request.constraint = ninfer::OutputConstraint::choice(cli.constraint.choices);
        else if (cli.constraint.kind == ninfer::product::ConstraintKind::Grammar ||
                 cli.constraint.kind == ninfer::product::ConstraintKind::JsonSchema) {
            const auto& path = cli.constraint.path;
            std::ifstream file(path, std::ios::binary);
            if (!file) throw std::invalid_argument("cannot open constraint file: " + path.string());
            std::string source(std::istreambuf_iterator<char>(file), {});
            if (file.bad()) throw std::runtime_error("failed to read constraint file");
            request.constraint = cli.constraint.kind == ninfer::product::ConstraintKind::JsonSchema
                                     ? ninfer::OutputConstraint::json_schema(std::move(source))
                                     : ninfer::OutputConstraint::grammar(std::move(source));
        }
        request.execution.sampling = defaults.sampling;
        if (!request.execution.sampling.seed)
            request.execution.sampling.seed = ninfer::product::random_sampling_seed();
        request.execution.requested_output_tokens = defaults.max_tokens;
        request.execution.thinking.budget         = defaults.thinking_budget;
        request.stop.token_ids                    = cli.stop_token_ids;
        request.stop.strings                      = cli.stop_strings;
        request.output.raw                        = cli.raw_output;

        auto engine_options             = cli.launch.engine;
        engine_options.startup_observer = startup_log.observer();

        ninfer::Engine engine(std::move(engine_options));
        startup_log.engine_ready(engine.startup_report().load);
        ninfer::product::log_resolved_configuration(*logger, engine.description());
        engine.reset_measurement_peaks();

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
        print_generation_summary(result, sampling, engine);
        return 0;
    } catch (const std::exception& error) {
        logger->error("{}", ninfer::product::format_pretty_text(error.what()));
        return 1;
    }
}
