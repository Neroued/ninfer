#include "serve/request_log.h"

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

std::uint64_t unix_time_ms() {
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(now).count());
}

std::string new_server_instance_id() {
    const auto now    = std::chrono::system_clock::now().time_since_epoch();
    const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    return "serve-" + std::to_string(static_cast<long long>(::getpid())) + '-' +
           std::to_string(micros);
}

std::filesystem::path normalized_absolute_path(const std::string& value) {
    std::error_code error;
    std::filesystem::path path = std::filesystem::weakly_canonical(value, error);
    if (!error) { return path; }
    error.clear();
    path = std::filesystem::absolute(value, error);
    return error ? std::filesystem::path(value).lexically_normal() : path.lexically_normal();
}

std::string cuda_version_string(int version) {
    if (version <= 0) { return {}; }
    return std::to_string(version / 1000) + '.' + std::to_string((version % 1000) / 10);
}

std::string cuda_uuid_string(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << "GPU-" << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) { out << '-'; }
        out << std::setw(2) << static_cast<unsigned int>(static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

const char* finish_reason_name(ninfer::FinishReason reason) {
    switch (reason) {
    case ninfer::FinishReason::None:
        return "none";
    case ninfer::FinishReason::OutputLimit:
        return "output_limit";
    case ninfer::FinishReason::ContextCapacity:
        return "context_capacity";
    case ninfer::FinishReason::StopToken:
        return "stop_token";
    case ninfer::FinishReason::StopString:
        return "stop_string";
    case ninfer::FinishReason::Cancelled:
        return "cancelled";
    }
    return "unknown";
}

std::string tool_choice_name(const ToolChoice& choice) {
    switch (choice.mode) {
    case ToolChoiceMode::Auto:
        return "auto";
    case ToolChoiceMode::None:
        return "none";
    case ToolChoiceMode::Required:
        return "required";
    case ToolChoiceMode::Named:
        return choice.name.empty() ? "named" : choice.name;
    }
    return "unknown";
}

const char* kv_cache_name(ninfer::KvCacheStorage storage) {
    return storage == ninfer::KvCacheStorage::BFloat16 ? "bf16" : "int8-group64";
}

const char* proposal_head_name(ninfer::ProposalHead proposal) {
    return proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full";
}

Json event_base(const std::string& server_instance_id, std::uint64_t timestamp, const char* event) {
    return Json{{"artifact_type", kRequestLogArtifactType},
                {"schema_version", kRequestLogSchemaVersion},
                {"event", event},
                {"timestamp_unix_ms", timestamp},
                {"server_instance_id", server_instance_id}};
}

Json sampler_json(const ninfer::SamplingParameters& sampling) {
    return Json{{"temperature", sampling.temperature},
                {"top_p", sampling.top_p},
                {"top_k", sampling.top_k},
                {"min_p", sampling.min_p},
                {"presence_penalty", sampling.presence_penalty},
                {"frequency_penalty", sampling.frequency_penalty},
                {"seed", sampling.seed}};
}

Json request_json(const RequestLogContext& context) {
    return Json{{"request_id", context.id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"enable_thinking", context.enable_thinking},
                {"sampling", sampler_json(context.sampling)}};
}

Json arena_json(const ninfer::ArenaMemorySummary& arena) {
    return Json{{"capacity_bytes", arena.capacity_bytes},
                {"used_bytes", arena.used_bytes},
                {"peak_used_bytes", arena.peak_used_bytes}};
}

Json mtp_json(const GenerationMetrics& metrics) {
    return Json{{"enabled", metrics.mtp_enabled},
                {"draft_window", metrics.mtp_draft_window},
                {"rounds", metrics.mtp_rounds},
                {"drafted_tokens", metrics.mtp_draft_tokens},
                {"accepted_tokens", metrics.mtp_accepted_tokens},
                {"fallback_steps", metrics.mtp_fallback_steps},
                {"accepted_per_position", metrics.mtp_accepted_per_position}};
}

// Tokens/second with fixed precision, or "n/a" when the interval is degenerate.
std::string rate(double tokens, double seconds) {
    std::ostringstream out;
    if (seconds > 0.0 && tokens > 0.0) {
        out << std::fixed << std::setprecision(1) << (tokens / seconds) << "tok/s";
    } else {
        out << "n/a";
    }
    return out.str();
}

std::string seconds_str(double seconds) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << seconds << 's';
    return out.str();
}

// Compact resolved-sampler summary. temperature <= 0 is the exact-argmax path.
std::string sampler_str(const ninfer::SamplingParameters& sampling) {
    if (sampling.temperature <= 0.0f) { return "greedy"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << "temp=" << sampling.temperature
        << " top_p=" << sampling.top_p << " top_k=" << sampling.top_k;
    if (sampling.min_p > 0.0f) { out << " min_p=" << sampling.min_p; }
    if (sampling.presence_penalty != 0.0f) { out << " pres=" << sampling.presence_penalty; }
    if (sampling.frequency_penalty != 0.0f) { out << " freq=" << sampling.frequency_penalty; }
    out << " seed=" << sampling.seed;
    return out.str();
}

std::string mtp_str(const GenerationMetrics& metrics) {
    if (!metrics.mtp_enabled) { return "off"; }
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (metrics.mtp_rounds > 0) {
        const double per_round = 1.0 + static_cast<double>(metrics.mtp_accepted_tokens) /
                                           static_cast<double>(metrics.mtp_rounds);
        out << per_round << "tok/round";
    } else {
        out << "n/a";
    }
    if (metrics.mtp_draft_tokens > 0) {
        const double accept_pct = 100.0 * static_cast<double>(metrics.mtp_accepted_tokens) /
                                  static_cast<double>(metrics.mtp_draft_tokens);
        out << " (" << std::setprecision(1) << accept_pct << "%)";
    }
    return out.str();
}

} // namespace

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const PreparedRequest& prepared) {
    RequestLogContext context;
    context.id                                 = id;
    context.protocol                           = std::move(protocol);
    context.model                              = request.model;
    context.stream                             = request.stream;
    context.message_count                      = request.messages.size();
    context.requested_output_tokens            = request.max_tokens;
    context.requested_output_tokens_client_set = request.max_tokens_set;
    context.tool_count                         = request.tools.size();
    context.tool_choice                        = request.tool_choice;
    context.has_tool_history                   = request.has_tool_history();
    context.enable_thinking                    = prepared.enable_thinking;
    context.sampling                           = prepared.options.execution.sampling;
    return context;
}

std::string format_request_start(const RequestLogContext& context) {
    std::ostringstream out;
    out << "[req " << context.id << "] " << context.protocol << ' '
        << (context.stream ? "stream" : "non-stream") << " msgs=" << context.message_count
        << " max_tokens=" << context.requested_output_tokens << ' '
        << (context.requested_output_tokens_client_set ? "(client)" : "(server default)")
        << " tools=" << context.tool_count
        << " tool_choice=" << tool_choice_name(context.tool_choice)
        << " tool_history=" << (context.has_tool_history ? "yes" : "no")
        << " thinking=" << (context.enable_thinking ? "on" : "off") << " sampler=["
        << sampler_str(context.sampling) << "] \xE2\x86\x92 running";
    return out.str();
}

std::string format_request_done(const RequestLogContext& context,
                                const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    const double ttft_ms =
        (metrics.prepare_seconds + metrics.vision_seconds + metrics.prefill_seconds) * 1000.0;
    // Prefill emits the first token; the remaining (gen - 1) come from decode.
    const double decode_tokens =
        outcome.completion_tokens > 0 ? static_cast<double>(outcome.completion_tokens - 1) : 0.0;

    std::ostringstream out;
    out << "[req " << context.id << "] done finish="
        << (outcome.tool_calls.empty() ? finish_reason_name(outcome.finish_reason) : "tool_calls");
    if (!outcome.tool_calls.empty()) { out << " tool_calls=" << outcome.tool_calls.size(); }
    out << " prompt=" << outcome.prompt_tokens << " gen=" << outcome.completion_tokens
        << " cache=" << metrics.prefix_cache_hit_tokens << " ttft=" << std::fixed
        << std::setprecision(0) << ttft_ms << "ms"
        << " prefill=" << rate(static_cast<double>(outcome.prompt_tokens), metrics.prefill_seconds)
        << " decode=" << rate(decode_tokens, metrics.decode_seconds)
        << " wall=" << seconds_str(metrics.total_seconds) << " mtp=" << mtp_str(metrics);
    return out.str();
}

std::string format_request_error(const RequestLogContext& context, const std::string& message) {
    std::ostringstream out;
    out << "[req " << context.id << "] error " << message;
    return out.str();
}

std::string format_server_start_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const ServeOptions& options, const ninfer::LoadSummary& load,
                                     const ninfer::MemorySummary& memory,
                                     const ServerLogEnvironment& environment,
                                     std::optional<std::uint64_t> artifact_size_bytes) {
    Json record = event_base(server_instance_id, timestamp, "server_start");

    Json default_seed = nullptr;
    if (options.sampling_seed.has_value()) { default_seed = *options.sampling_seed; }
    Json artifact_size = nullptr;
    if (artifact_size_bytes.has_value()) { artifact_size = *artifact_size_bytes; }

    record["server"]            = Json{{"host", options.host},
                                       {"port", options.port},
                                       {"public_model_id", options.model_id},
                                       {"api_key_configured", !options.api_key.empty()},
                                       {"cors_enabled", options.enable_cors},
                                       {"max_request_bytes", options.max_request_bytes},
                                       {"request_log_jsonl", options.request_log_jsonl},
                                       {"default_output_tokens", options.default_max_tokens},
                                       {"default_thinking", options.enable_thinking}};
    record["artifact"]          = Json{{"path", options.artifact_path},
                                       {"size_bytes", std::move(artifact_size)},
                                       {"target", load.target},
                                       {"bytes_read", load.artifact_bytes_read},
                                       {"host_to_device_bytes", load.host_to_device_bytes},
                                       {"peak_staging_bytes", load.peak_staging_bytes},
                                       {"tensor_count", load.tensor_count},
                                       {"resource_count", load.resource_count},
                                       {"load_seconds", load.load_seconds},
                                       {"upload_seconds", load.upload_seconds}};
    record["engine"]            = Json{{"device", options.device},
                                       {"max_context", options.max_context},
                                       {"prefill_chunk", options.prefill_chunk},
                                       {"kv_cache", kv_cache_name(options.kv_cache)},
                                       {"cuda_graph", options.use_cuda_graph},
                                       {"prefix_reuse", options.allow_prefix_reuse},
                                       {"mtp_draft_window", options.mtp_draft_tokens},
                                       {"mtp_proposal_head", proposal_head_name(options.proposal_head)}};
    record["sampling_defaults"] = Json{{"temperature", options.sampling_temperature},
                                       {"top_p", options.sampling_top_p},
                                       {"top_k", options.sampling_top_k},
                                       {"min_p", 0.0},
                                       {"presence_penalty", options.sampling_presence_penalty},
                                       {"frequency_penalty", options.sampling_frequency_penalty},
                                       {"seed", std::move(default_seed)},
                                       {"greedy", options.greedy}};
    record["memory"]            = Json{{"weights", arena_json(memory.weights)},
                                       {"sequence", arena_json(memory.sequence)},
                                       {"workspace", arena_json(memory.workspace)},
                                       {"kv_payload_bytes", memory.kv_payload_bytes}};
    record["environment"] =
        Json{{"device", environment.device},
             {"gpu_name", environment.gpu_name},
             {"gpu_uuid", environment.gpu_uuid},
             {"total_device_memory_bytes", environment.total_device_memory_bytes},
             {"compute_capability_major", environment.compute_capability_major},
             {"compute_capability_minor", environment.compute_capability_minor},
             {"cuda_compile_version", environment.cuda_compile_version},
             {"cuda_runtime_version", environment.cuda_runtime_version},
             {"cuda_driver_version", environment.cuda_driver_version}};
    record["argv"] = options.startup_argv;
    return record.dump();
}

std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "request_start");
    record["request"] = request_json(context);
    return record.dump();
}

std::string format_request_done_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    Json record               = event_base(server_instance_id, timestamp, "request_done");
    record["request"]         = request_json(context);
    record["result"]          = Json{{"finish_reason", finish_reason_name(outcome.finish_reason)},
                                     {"prompt_tokens", outcome.prompt_tokens},
                                     {"completion_tokens", outcome.completion_tokens},
                                     {"prefix_cache_hit_tokens", outcome.metrics.prefix_cache_hit_tokens},
                                     {"tool_call_count", outcome.tool_calls.size()}};
    record["timings_seconds"] = Json{{"prepare", outcome.metrics.prepare_seconds},
                                     {"vision", outcome.metrics.vision_seconds},
                                     {"prefill", outcome.metrics.prefill_seconds},
                                     {"decode", outcome.metrics.decode_seconds},
                                     {"total", outcome.metrics.total_seconds}};
    record["mtp"]             = mtp_json(outcome.metrics);
    return record.dump();
}

std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context,
                                      const std::string& message) {
    Json record       = event_base(server_instance_id, timestamp, "request_error");
    record["request"] = request_json(context);
    record["error"]   = Json{{"message", message}};
    return record.dump();
}

ServerLogEnvironment query_server_log_environment(int device) {
    ServerLogEnvironment environment;
    environment.device               = device;
    environment.cuda_compile_version = cuda_version_string(CUDART_VERSION);

    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) == cudaSuccess) {
        environment.cuda_runtime_version = cuda_version_string(runtime_version);
    }
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        environment.cuda_driver_version = cuda_version_string(driver_version);
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        environment.gpu_name                  = properties.name;
        environment.gpu_uuid                  = cuda_uuid_string(properties.uuid);
        environment.total_device_memory_bytes = properties.totalGlobalMem;
        environment.compute_capability_major  = properties.major;
        environment.compute_capability_minor  = properties.minor;
    }
    return environment;
}

JsonlRequestLog::JsonlRequestLog(const std::string& path,
                                 const std::string& protected_artifact_path)
    : path_(path) {
    if (path_.empty()) { return; }
    if (!protected_artifact_path.empty() &&
        normalized_absolute_path(path_) == normalized_absolute_path(protected_artifact_path)) {
        throw std::invalid_argument("request JSONL log must not overwrite the model artifact");
    }
    server_instance_id_ = new_server_instance_id();
    output_.open(path_, std::ios::out | std::ios::app);
    if (!output_) {
        throw std::runtime_error("failed to open request JSONL log for append: " + path_);
    }
}

void JsonlRequestLog::write_server_start(const ServeOptions& options,
                                         const ninfer::LoadSummary& load,
                                         const ninfer::MemorySummary& memory) {
    if (!enabled()) { return; }
    std::error_code error;
    const std::uintmax_t size = std::filesystem::file_size(options.artifact_path, error);
    const std::optional<std::uint64_t> artifact_size =
        error ? std::nullopt : std::optional<std::uint64_t>(size);
    append(format_server_start_json(server_instance_id_, unix_time_ms(), options, load, memory,
                                    query_server_log_environment(options.device), artifact_size));
}

void JsonlRequestLog::write_request_start(const RequestLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_start_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_done(const RequestLogContext& context,
                                         const GenerationOutcome& outcome) {
    if (!enabled()) { return; }
    append(format_request_done_json(server_instance_id_, unix_time_ms(), context, outcome));
}

void JsonlRequestLog::write_request_error(const RequestLogContext& context,
                                          const std::string& message) {
    if (!enabled()) { return; }
    append(format_request_error_json(server_instance_id_, unix_time_ms(), context, message));
}

void JsonlRequestLog::append(std::string record) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (failed_) { return; }
    output_ << record << '\n';
    output_.flush();
    if (!output_) {
        failed_ = true;
        std::cerr << "ninfer-serve: request JSONL logging failed for " << path_ << '\n';
    }
}

} // namespace ninfer::serve
