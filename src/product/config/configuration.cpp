#include "product/config/schema.h"

#include "product/speculative_options.h"

#include <cstdlib>
#include <type_traits>

namespace ninfer::product {
namespace {

using namespace config_detail;

template <class T>
ConfigurationValue optional_value(const std::optional<T>& value, const char* fallback = "model") {
    if (!value) return std::string(fallback);
    if constexpr (std::is_same_v<T, bool>)
        return *value;
    else if constexpr (std::is_floating_point_v<T>)
        return static_cast<double>(*value);
    else
        return static_cast<std::uint64_t>(*value);
}

const char* kv_name(KvCacheStorage storage) {
    switch (storage) {
    case KvCacheStorage::BFloat16:
        return "bf16";
    case KvCacheStorage::Int8Group64:
        return "int8";
    case KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

const char* effort_name(ReasoningEffort effort) {
    switch (effort) {
    case ReasoningEffort::None:
        return "none";
    case ReasoningEffort::Minimal:
        return "minimal";
    case ReasoningEffort::Low:
        return "low";
    case ReasoningEffort::Medium:
        return "medium";
    case ReasoningEffort::High:
        return "high";
    case ReasoningEffort::XHigh:
        return "xhigh";
    case ReasoningEffort::Max:
        return "max";
    }
    return "unknown";
}

const char* constraint_name(ConstraintKind kind) {
    switch (kind) {
    case ConstraintKind::None:
        return "none";
    case ConstraintKind::Grammar:
        return "grammar";
    case ConstraintKind::JsonObject:
        return "json_object";
    case ConstraintKind::JsonSchema:
        return "json_schema";
    case ConstraintKind::Regex:
        return "regex";
    case ConstraintKind::Choice:
        return "choice";
    }
    return "unknown";
}

const char* log_level_name(LogLevel level) {
    switch (level) {
    case LogLevel::Trace:
        return "trace";
    case LogLevel::Debug:
        return "debug";
    case LogLevel::Info:
        return "info";
    case LogLevel::Warning:
        return "warning";
    case LogLevel::Error:
        return "error";
    case LogLevel::Critical:
        return "critical";
    case LogLevel::Off:
        return "off";
    }
    return "unknown";
}

StartupConfigurationView view(const LaunchOptions& launch, const CliOptions* cli,
                              const ServeOptions* server) {
    const auto& e = launch.engine;
    const auto& g = launch.generation;
    const auto& s = g.sampling;
    StartupConfigurationView out{.entry       = cli ? EntryPoint::Cli : EntryPoint::Serve,
                                 .config_path = launch.provenance.config_path,
                                 .argv        = launch.provenance.argv};
    for (const auto& spec : fields()) {
        if (!applies_to(spec, out.entry)) continue;
        ConfigurationValue value;
        switch (spec.field) {
        case Field::ModelPath:
            value = e.artifact_path.string();
            break;
        case Field::ChatTemplate:
            value = e.chat_template_path.string();
            break;
        case Field::Vision:
            value = e.enable_vision;
            break;
        case Field::Device:
            value = static_cast<std::uint64_t>(e.device);
            break;
        case Field::MaxContext:
            value = std::uint64_t{e.max_context};
            break;
        case Field::KvDtype:
            value = std::string(kv_name(e.kv_cache));
            break;
        case Field::KvCapacity:
            if (e.kv_capacity.mode == KvCapacityMode::Automatic)
                value = std::string("auto");
            else
                value = std::uint64_t{e.kv_capacity.explicit_tokens};
            break;
        case Field::PrefillChunk:
            if (e.prefill_chunk == 0)
                value = std::string("default (model/placement)");
            else
                value = std::uint64_t{e.prefill_chunk};
            break;
        case Field::CudaGraph:
            value = e.use_cuda_graph;
            break;
        case Field::MaxConcurrency:
            value = std::uint64_t{e.max_concurrency};
            break;
        case Field::MaxPendingRequests:
            value = std::uint64_t{e.max_pending_requests};
            break;
        case Field::PendingTimeout:
            value = std::uint64_t{e.pending_timeout_ms};
            break;
        case Field::PrefixReuse:
            value = e.context_cache.enabled;
            break;
        case Field::ExtraStateSlots:
            value = std::uint64_t{e.context_cache.extra_state_slots.value_or(e.max_concurrency)};
            break;
        case Field::HostCapacity:
            value = optional_value(e.context_cache.host_capacity_bytes, "default (target/model)");
            break;
        case Field::CostPresets:
            value = e.context_cost.preset_path.string();
            break;
        case Field::SpecBackend:
            value = std::string(speculative_backend_name(e.speculative.backend));
            break;
        case Field::DraftTokens:
            if (e.speculative.backend == SpeculativeBackend::None) continue;
            value = std::uint64_t{e.speculative.draft_tokens};
            break;
        case Field::ProposalHead:
            if (e.speculative.backend == SpeculativeBackend::None) continue;
            value = std::string(e.speculative.proposal_head == ProposalHead::Full ? "full"
                                                                                  : "optimized");
            break;
        case Field::MediaCache:
            value = std::uint64_t{e.media_cache_bytes};
            break;
        case Field::MediaLive:
            value = std::uint64_t{e.media_live_bytes};
            break;
        case Field::MediaThreads:
            if (e.media_preprocess_threads == 0)
                value = std::string("auto");
            else
                value = std::uint64_t{e.media_preprocess_threads};
            break;
        case Field::MaxTokens:
            value = std::uint64_t{g.max_tokens};
            break;
        case Field::Thinking:
            value = optional_value(g.enable_thinking);
            break;
        case Field::PreserveThinking:
            value = optional_value(g.preserve_thinking);
            break;
        case Field::ThinkingBudget:
            value = optional_value(g.thinking_budget, "none");
            break;
        case Field::Temperature:
            value = optional_value(s.temperature);
            break;
        case Field::TopK:
            value = optional_value(s.top_k);
            break;
        case Field::TopP:
            value = optional_value(s.top_p);
            break;
        case Field::MinP:
            value = optional_value(s.min_p);
            break;
        case Field::PresencePenalty:
            value = optional_value(s.presence_penalty);
            break;
        case Field::FrequencyPenalty:
            value = optional_value(s.frequency_penalty);
            break;
        case Field::Seed:
            value = optional_value(s.seed, "random per request");
            break;
        case Field::ServerHost:
            value = server->host;
            break;
        case Field::ServerPort:
            value = static_cast<std::uint64_t>(server->port);
            break;
        case Field::ApiKey:
            value = std::string(server->api_key.empty() ? "disabled" : "configured");
            break;
        case Field::ModelId:
            value = server->model_id_override.value_or("artifact name");
            break;
        case Field::Cors:
            value = server->enable_cors;
            break;
        case Field::MaxRequestSize:
            value = std::uint64_t{server->max_request_bytes};
            break;
        case Field::ResponseRecords:
            value = std::uint64_t{server->response_store_max_records};
            break;
        case Field::ResponseSize:
            value = std::uint64_t{server->response_store_max_bytes};
            break;
        case Field::LogLevel:
            value = std::string(log_level_name(launch.logging.level));
            break;
        case Field::StatsInterval:
            value = std::uint64_t{launch.logging.stats_interval_ms};
            break;
        case Field::RequestJsonl:
            value = launch.logging.request_jsonl.string();
            break;
        case Field::Prompt:
            if (!cli->messages_path.empty()) continue;
            value = cli->prompt;
            break;
        case Field::Messages:
            if (cli->messages_path.empty()) continue;
            value = cli->messages_path.string();
            break;
        case Field::ReasoningEffort:
            value =
                std::string(cli->reasoning_effort ? effort_name(*cli->reasoning_effort) : "model");
            break;
        case Field::RawOutput:
            value = cli->raw_output;
            break;
        case Field::PrintTokenIds:
            value = cli->print_token_ids;
            break;
        case Field::StopIds:
            value =
                std::vector<std::uint32_t>(cli->stop_token_ids.begin(), cli->stop_token_ids.end());
            break;
        case Field::Stop:
        case Field::ReasoningStop: {
            std::vector<std::string> stops;
            const auto channel =
                spec.field == Field::Stop ? OutputChannel::Content : OutputChannel::Reasoning;
            for (const auto& stop : cli->stop_strings)
                if (stop.channel == channel) stops.push_back(stop.text);
            value = std::move(stops);
            break;
        }
        case Field::ConstraintKind:
            value = std::string(constraint_name(cli->constraint.kind));
            break;
        case Field::GrammarFile:
            if (cli->constraint.kind != ConstraintKind::Grammar) continue;
            value = cli->constraint.path.string();
            break;
        case Field::JsonSchemaFile:
            if (cli->constraint.kind != ConstraintKind::JsonSchema) continue;
            value = cli->constraint.path.string();
            break;
        case Field::Regex:
            if (cli->constraint.kind != ConstraintKind::Regex) continue;
            value = cli->constraint.regex;
            break;
        case Field::Choices:
            if (cli->constraint.kind != ConstraintKind::Choice) continue;
            value = cli->constraint.choices;
            break;
        case Field::Count:
            continue;
        }
        const auto source = launch.provenance.fields.find(spec.key);
        out.items.push_back(
            {std::string(spec.key), std::move(value),
             source == launch.provenance.fields.end() ? ConfigSource{} : source->second,
             spec.unit});
    }
    if (e.deployment) {
        const auto append = [&](std::string key, ConfigurationValue value,
                                ConfigurationUnit unit = ConfigurationUnit::None) {
            const auto source = launch.provenance.fields.find(key);
            out.items.push_back(
                {std::move(key), std::move(value),
                 source == launch.provenance.fields.end() ? ConfigSource{} : source->second, unit});
        };
        const auto& d = *e.deployment;
        append("distributed.coordinator_node", std::uint64_t{d.coordinator_node});
        append("distributed.control_host", d.control_host);
        append("distributed.control_port", std::uint64_t{d.control_port});
        append("distributed.connect_timeout", std::uint64_t{d.connect_timeout_ms},
               ConfigurationUnit::Milliseconds);
        append("distributed.shutdown_timeout", std::uint64_t{d.shutdown_timeout_ms},
               ConfigurationUnit::Milliseconds);
        append("distributed.parallel.tp", std::uint64_t{d.parallel.tp});
        append("distributed.parallel.ep", std::uint64_t{d.parallel.ep});
        append("distributed.parallel.dp", std::uint64_t{d.parallel.dp});
        append("distributed.parallel.pp", std::uint64_t{d.parallel.pp});
        for (std::size_t i = 0; i < d.nodes.size(); ++i) {
            const auto key   = "distributed.nodes[" + std::to_string(i) + "].";
            const auto& node = d.nodes[i];
            append(key + "id", std::uint64_t{node.id});
            append(key + "ssh", node.ssh);
            append(key + "worker", node.worker.string());
            append(key + "artifact", node.artifact.string());
            for (const auto& [name, value] : node.environment)
                append(key + "environment." + name, value);
        }
        for (std::size_t i = 0; i < d.ranks.size(); ++i) {
            const auto key   = "distributed.ranks[" + std::to_string(i) + "].";
            const auto& rank = d.ranks[i];
            append(key + "rank", std::uint64_t{rank.rank});
            append(key + "node", std::uint64_t{rank.node});
            append(key + "device", static_cast<std::uint64_t>(rank.device));
        }
    }
    for (const char* name : {"NINFER_VRAM_BUDGET_MIB", "NINFER_CUDA_SYNC"}) {
        if (const char* value = std::getenv(name)) out.environment.emplace(name, value);
    }
    return out;
}

} // namespace

StartupConfigurationView configuration_view(const CliOptions& options) {
    return view(options.launch, &options, nullptr);
}

StartupConfigurationView configuration_view(const ServeOptions& options) {
    return view(options.launch, nullptr, &options);
}

} // namespace ninfer::product
