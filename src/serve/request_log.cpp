#include "serve/request_log.h"
#include "product/statistics.h"
#include "product/constraint_observation.h"
#include "product/logging/pretty_format.h"
#include "product/speculative_options.h"

#include <spdlog/logger.h>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#include <unistd.h>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

template <class T>
T monotonic_delta(T previous, T current) noexcept {
    return current >= previous ? current - previous : T{};
}

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

Json tool_call_parse_json(const ninfer::ToolCallParseDiagnostics& diagnostics) {
    return Json{{"marker_seen", diagnostics.marker_seen},
                {"structured_call_count", diagnostics.structured_call_count},
                {"schema_mismatch_arguments", diagnostics.schema_mismatch_arguments},
                {"fallback_reason",
                 ninfer::tool_call_parse_fallback_reason_name(diagnostics.fallback_reason)}};
}

std::string tool_choice_name(const ToolChoice& choice) {
    switch (choice.mode) {
    case ToolChoiceMode::Auto:
        return "auto";
    case ToolChoiceMode::None:
        return "none";
    }
    return "unknown";
}

Json requested_reasoning_effort_json(const std::optional<RequestedReasoningEffort>& requested) {
    return requested ? Json(std::string(requested_reasoning_effort_name(*requested)))
                     : Json(nullptr);
}

const char* kv_cache_name(ninfer::KvCacheStorage storage) {
    return ninfer::kv_cache_profile_info(storage).name;
}

const char* context_capacity_mode_name(ninfer::ContextCapacityMode mode) {
    return mode == ninfer::ContextCapacityMode::Automatic ? "auto" : "explicit";
}

const char* proposal_head_name(ninfer::ProposalHead proposal) {
    return proposal == ninfer::ProposalHead::Optimized ? "optimized" : "full";
}

const char* prefix_reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
    case ninfer::PrefixReusePath::Root:
        return "root";
    case ninfer::PrefixReusePath::Checkpoint:
        return "checkpoint";
    }
    return "unknown";
}

Json event_base(const std::string& server_instance_id, std::uint64_t timestamp, const char* event) {
    return Json{{"artifact_type", kRequestLogArtifactType},
                {"schema_version", kRequestLogSchemaVersion},
                {"event", event},
                {"timestamp_unix_ms", timestamp},
                {"server_instance_id", server_instance_id}};
}

const char* scheduling_transition_name(ninfer::GenerationSchedulingTransition transition) {
    using Transition = ninfer::GenerationSchedulingTransition;
    switch (transition) {
    case Transition::PauseStarted:
        return "pause_started";
    case Transition::Paused:
        return "paused";
    case Transition::RestoreStarted:
        return "restore_started";
    case Transition::Restored:
        return "restored";
    case Transition::ReplayComplete:
        return "replay_complete";
    case Transition::RecoveryComplete:
        return "recovery_complete";
    case Transition::SnapshotRevoked:
        return "snapshot_revoked";
    case Transition::Terminal:
        return "terminal";
    }
    throw std::logic_error("invalid scheduling transition");
}

Json recovery_route_json(ninfer::GenerationRecoveryRoute route) {
    switch (route) {
    case ninfer::GenerationRecoveryRoute::None:
        return nullptr;
    case ninfer::GenerationRecoveryRoute::Snapshot:
        return "snapshot";
    case ninfer::GenerationRecoveryRoute::Replay:
        return "replay";
    }
    throw std::logic_error("invalid recovery route");
}

Json sampler_json(const ninfer::ResolvedSamplingParameters& sampling) {
    return Json{{"temperature", sampling.temperature},
                {"top_p", sampling.top_p},
                {"top_k", sampling.top_k},
                {"min_p", sampling.min_p},
                {"presence_penalty", sampling.presence_penalty},
                {"frequency_penalty", sampling.frequency_penalty},
                {"seed", sampling.seed}};
}

Json preset_json(const ninfer::SamplingPreset& preset) {
    return Json{{"temperature", preset.temperature},
                {"top_p", preset.top_p},
                {"top_k", preset.top_k},
                {"min_p", preset.min_p},
                {"presence_penalty", preset.presence_penalty},
                {"frequency_penalty", preset.frequency_penalty}};
}

Json overrides_json(const ninfer::SamplingOverrides& overrides) {
    Json result{{"temperature", nullptr},
                {"top_p", nullptr},
                {"top_k", nullptr},
                {"min_p", nullptr},
                {"presence_penalty", nullptr},
                {"frequency_penalty", nullptr},
                {"seed", nullptr}};
    if (overrides.temperature) { result["temperature"] = *overrides.temperature; }
    if (overrides.top_p) { result["top_p"] = *overrides.top_p; }
    if (overrides.top_k) { result["top_k"] = *overrides.top_k; }
    if (overrides.min_p) { result["min_p"] = *overrides.min_p; }
    if (overrides.presence_penalty) { result["presence_penalty"] = *overrides.presence_penalty; }
    if (overrides.frequency_penalty) { result["frequency_penalty"] = *overrides.frequency_penalty; }
    if (overrides.seed) { result["seed"] = *overrides.seed; }
    return result;
}

Json request_json(const RequestLogContext& context) {
    Json thinking_budget = nullptr;
    if (context.thinking_budget) { thinking_budget = *context.thinking_budget; }
    return Json{{"request_id", context.id},
                {"http_request_id", context.http_request_id},
                {"response_id", context.response_id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"enable_thinking", context.enable_thinking},
                {"thinking_budget", std::move(thinking_budget)},
                {"requested_reasoning_effort",
                 requested_reasoning_effort_json(context.requested_reasoning_effort)},
                {"preserve_thinking",
                 context.preserve_thinking ? Json(*context.preserve_thinking) : Json(nullptr)},
                {"preserve_thinking_semantic_change", context.preserve_thinking_semantic_change},
                {"sampling", sampler_json(context.sampling)}};
}

Json preparation_json(const RequestLogContext& context) {
    const PromptPreparationStats& stats = context.preparation;
    return Json{{"total", stats.seconds},
                {"acquisition", context.acquisition_seconds},
                {"media_preprocess", stats.media_preprocess_seconds},
                {"media_preprocess_work", stats.media_preprocess_work_seconds},
                {"tokenize", stats.tokenize_seconds},
                {"media_items", stats.media_items},
                {"media_bytes", stats.media_bytes},
                {"raw_patches", stats.raw_patches},
                {"vision_tokens", stats.vision_tokens},
                {"patch_bytes", stats.patch_bytes},
                {"cache_hits", stats.media_cache_hits},
                {"cache_misses", stats.media_cache_misses},
                {"singleflight_waits", stats.media_singleflight_waits},
                {"built_patch_bytes", stats.built_patch_bytes},
                {"reused_patch_bytes", stats.reused_patch_bytes}};
}

Json rejected_request_json(const RequestRejectionLogContext& context) {
    return Json{{"request_id", context.id},
                {"http_request_id", context.http_request_id},
                {"response_id", context.response_id},
                {"protocol", context.protocol},
                {"model", context.model},
                {"stream", context.stream},
                {"message_count", context.message_count},
                {"media_item_count", context.media_item_count},
                {"requested_output_tokens", context.requested_output_tokens},
                {"requested_output_tokens_source",
                 context.requested_output_tokens_client_set ? "client" : "server_default"},
                {"tool_count", context.tool_count},
                {"tool_choice", tool_choice_name(context.tool_choice)},
                {"has_tool_history", context.has_tool_history},
                {"requested_reasoning_effort",
                 requested_reasoning_effort_json(context.requested_reasoning_effort)}};
}

Json error_json(const ApiError& error) {
    Json code  = error.code.empty() ? Json(nullptr) : Json(error.code);
    Json param = error.param.empty() ? Json(nullptr) : Json(error.param);
    return Json{{"status", error.status},
                {"type", error.type},
                {"code", std::move(code)},
                {"param", std::move(param)},
                {"message", error.message}};
}

Json speculative_json(const RequestStats& metrics) {
    return Json{{"backend", product::speculative_backend_name(metrics.speculative.backend)},
                {"draft_window", metrics.speculative.draft_window},
                {"rounds", metrics.speculative.rounds},
                {"drafted_tokens", metrics.speculative.drafted_tokens},
                {"accepted_tokens", metrics.speculative.accepted_tokens},
                {"committed_draft_tokens", metrics.speculative.committed_draft_tokens},
                {"fallback_steps", metrics.speculative.fallback_steps},
                {"accepted_per_position", metrics.speculative.accepted_per_position}};
}

Json scheduling_json(const ninfer::GenerationSchedulingStats& stats) {
    return Json{{"preemptions", stats.preemptions},
                {"snapshot_restores", stats.snapshot_restores},
                {"replay_restores", stats.replay_restores},
                {"replayed_tokens", stats.replayed_tokens},
                {"paused_ns", stats.paused_ns},
                {"device_to_host_bytes", stats.device_to_host_bytes},
                {"host_to_device_bytes", stats.host_to_device_bytes}};
}

Json admission_json(const ninfer::GenerationAdmissionStats& stats) {
    const char* reason = nullptr;
    switch (stats.fallback_reason) {
    case ninfer::AdmissionFallbackReason::None:
        reason = "none";
        break;
    case ninfer::AdmissionFallbackReason::SourceInvalid:
        reason = "source_invalid";
        break;
    case ninfer::AdmissionFallbackReason::SourceRevoked:
        reason = "source_revoked";
        break;
    case ninfer::AdmissionFallbackReason::CostChanged:
        reason = "cost_changed";
        break;
    case ninfer::AdmissionFallbackReason::CapacityLimit:
        reason = "capacity_limit";
        break;
    case ninfer::AdmissionFallbackReason::IsolatedCapacity:
        reason = "isolated_capacity";
        break;
    }
    if (!reason) { throw std::logic_error("invalid admission fallback reason"); }
    return Json{{"preferred_reused_tokens", stats.preferred_reused_tokens},
                {"source_wait_seconds", stats.source_wait_seconds},
                {"revoked_checkpoints", stats.revoked_checkpoints},
                {"fallback_reason", reason}};
}

double nanoseconds_to_seconds(std::uint64_t value) noexcept {
    return static_cast<double>(value) * 1.0e-9;
}

double nanoseconds_to_microseconds(std::uint64_t value) noexcept {
    return static_cast<double>(value) * 1.0e-3;
}

double request_host_exposed_seconds(const ninfer::GenerationEngineTiming& timing) noexcept {
    return timing.engine_boundary_exposed_seconds + timing.session_call_exposed_seconds +
           timing.executor_poll_exposed_seconds + timing.engine_commit_output_exposed_seconds +
           timing.engine_maintenance_exposed_seconds + timing.executor_flush_exposed_seconds;
}

Json request_engine_timing_json(const ninfer::GenerationEngineTiming& timing) {
    return Json{
        {"queue_wait_seconds", timing.queue_wait_seconds},
        {"host_exposed_seconds",
         Json{{"engine_boundary", timing.engine_boundary_exposed_seconds},
              {"session_call", timing.session_call_exposed_seconds},
              {"executor_poll", timing.executor_poll_exposed_seconds},
              {"executor_flush", timing.executor_flush_exposed_seconds},
              {"engine_commit_output", timing.engine_commit_output_exposed_seconds},
              {"engine_maintenance", timing.engine_maintenance_exposed_seconds},
              {"total", request_host_exposed_seconds(timing)}}},
        {"scheduler_yield_exposed_seconds", timing.scheduler_yield_exposed_seconds},
        {"decode",
         Json{{"host_exposed_seconds", timing.decode_host_exposed_seconds},
              {"scheduler_yield_exposed_seconds", timing.decode_scheduler_yield_exposed_seconds},
              {"rounds", timing.decode_rounds}}},
        {"units", Json{{"prefill", timing.prefill_units}, {"control", timing.control_units}}},
    };
}

Json transfer_timings_json(const ninfer::ContextTransferSeries& current,
                           const ninfer::ContextTransferSeries& previous = {}) {
    Json result = Json::object();
    constexpr std::array directions{"d2h", "h2d", "d2d", "h2h", "peer_d2d"};
    for (std::size_t i = 0; i < directions.size(); ++i) {
        const auto& now       = current[i];
        const auto& before    = previous[i];
        result[directions[i]] = {
            {"images", monotonic_delta(before.images, now.images)},
            {"page_groups", monotonic_delta(before.page_groups, now.page_groups)},
            {"bytes", monotonic_delta(before.bytes, now.bytes)},
            {"copy_operations", monotonic_delta(before.copy_operations, now.copy_operations)},
            {"seconds", monotonic_delta(before.seconds, now.seconds)}};
    }
    return result;
}

Json first_output_timing_json(const ninfer::GenerationFirstOutputTiming& timing) {
    Json pools = Json::array();
    for (std::size_t i = 0; i < timing.context_transfers.kv.size(); ++i) {
        pools.push_back(
            {{"pool_id", i}, {"transfers", transfer_timings_json(timing.context_transfers.kv[i])}});
    }
    Json transfers{{"state", transfer_timings_json(timing.context_transfers.state)},
                   {"kv_pools", std::move(pools)}};
    return Json{{"elapsed_seconds", timing.elapsed_seconds},
                {"initial_binding_seconds", timing.initial_binding_seconds},
                {"engine", request_engine_timing_json(timing.engine)},
                {"prefill", product::execution_json(timing.prefill)},
                {"replay", product::execution_json(timing.replay)},
                {"pending_execution", timing.pending_execution
                                          ? product::execution_json(*timing.pending_execution)
                                          : Json(nullptr)},
                {"scheduling", scheduling_json(timing.scheduling)},
                {"computed_prefill_tokens", timing.computed_prefill_tokens},
                {"context_transfers", std::move(transfers)}};
}

ninfer::RuntimeHostWorkStats host_work_delta(const ninfer::RuntimeHostWorkStats& previous,
                                             const ninfer::RuntimeHostWorkStats& current) {
    return ninfer::RuntimeHostWorkStats{
        .engine_boundary_ns =
            monotonic_delta(previous.engine_boundary_ns, current.engine_boundary_ns),
        .session_call_ns  = monotonic_delta(previous.session_call_ns, current.session_call_ns),
        .executor_poll_ns = monotonic_delta(previous.executor_poll_ns, current.executor_poll_ns),
        .engine_commit_output_ns =
            monotonic_delta(previous.engine_commit_output_ns, current.engine_commit_output_ns),
        .engine_maintenance_ns =
            monotonic_delta(previous.engine_maintenance_ns, current.engine_maintenance_ns),
        .scheduler_yield_ns =
            monotonic_delta(previous.scheduler_yield_ns, current.scheduler_yield_ns),
        .executor_flush_ns = monotonic_delta(previous.executor_flush_ns, current.executor_flush_ns),
        .decode_host_ns    = monotonic_delta(previous.decode_host_ns, current.decode_host_ns),
        .decode_scheduler_yield_ns =
            monotonic_delta(previous.decode_scheduler_yield_ns, current.decode_scheduler_yield_ns),
        .prefill_host_ns = monotonic_delta(previous.prefill_host_ns, current.prefill_host_ns),
        .prefill_scheduler_yield_ns = monotonic_delta(previous.prefill_scheduler_yield_ns,
                                                      current.prefill_scheduler_yield_ns),
        .control_host_ns = monotonic_delta(previous.control_host_ns, current.control_host_ns),
        .control_scheduler_yield_ns = monotonic_delta(previous.control_scheduler_yield_ns,
                                                      current.control_scheduler_yield_ns),
        .prefill_units = monotonic_delta(previous.prefill_units, current.prefill_units),
        .control_units = monotonic_delta(previous.control_units, current.control_units),
        .stats_publication_ns =
            monotonic_delta(previous.stats_publication_ns, current.stats_publication_ns),
        .stats_publication_invocations = monotonic_delta(previous.stats_publication_invocations,
                                                         current.stats_publication_invocations),
    };
}

std::uint64_t host_active_ns(const ninfer::RuntimeHostWorkStats& timing) noexcept {
    return timing.engine_boundary_ns + timing.session_call_ns + timing.executor_poll_ns +
           timing.engine_commit_output_ns + timing.engine_maintenance_ns + timing.executor_flush_ns;
}

Json microseconds_per(std::uint64_t nanoseconds, std::uint64_t count) {
    if (count == 0) { return nullptr; }
    return nanoseconds_to_microseconds(nanoseconds) / static_cast<double>(count);
}

} // namespace

std::string format_startup_config_json(const std::string& server_instance_id,
                                       std::uint64_t timestamp,
                                       const product::ServeOptions& options) {
    const auto view         = product::configuration_view(options);
    Json record             = event_base(server_instance_id, timestamp, "startup_config");
    record["entry"]         = "serve";
    record["config_path"]   = view.config_path.string();
    record["argv"]          = view.argv;
    record["environment"]   = view.environment;
    record["configuration"] = Json::object();
    record["sources"]       = Json::object();
    for (const auto& item : view.items) {
        record["configuration"][item.key] =
            std::visit([](const auto& value) { return Json(value); }, item.value);
        Json source;
        switch (item.source.kind) {
        case product::ConfigSourceKind::Default:
            source = {{"kind", "default"}};
            break;
        case product::ConfigSourceKind::File:
            source = {{"kind", "file"}, {"line", item.source.line}, {"column", item.source.column}};
            break;
        case product::ConfigSourceKind::CommandLine:
            source = {{"kind", "argv"}, {"argument", item.source.argument}};
            break;
        }
        record["sources"][item.key] = std::move(source);
    }
    return record.dump();
}

std::string format_server_start_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const product::ServeOptions& options,
                                     const ninfer::EngineOptions& engine_options,
                                     const ninfer::ModelSamplingDefaults& sampling_defaults,
                                     const std::string& public_model_id,
                                     const EngineDescription& description,
                                     const StartupReport& startup, const EngineSnapshot& snapshot,
                                     const ServerLogEnvironment& environment,
                                     std::optional<std::uint64_t> artifact_size_bytes) {
    Json record      = event_base(server_instance_id, timestamp, "server_start");
    const auto& load = startup.load;

    Json artifact_size = nullptr;
    if (artifact_size_bytes.has_value()) { artifact_size = *artifact_size_bytes; }
    Json default_thinking_budget = nullptr;
    if (options.launch.generation.thinking_budget) {
        default_thinking_budget = *options.launch.generation.thinking_budget;
    }

    record["server"] =
        Json{{"host", options.host},
             {"port", options.port},
             {"public_model_id", public_model_id},
             {"api_key_configured", !options.api_key.empty()},
             {"cors_enabled", options.enable_cors},
             {"max_request_bytes", options.max_request_bytes},
             {"media_cache_bytes", options.launch.engine.media_cache_bytes},
             {"media_live_bytes", options.launch.engine.media_live_bytes},
             {"media_preprocess_threads", options.launch.engine.media_preprocess_threads},
             {"request_log_jsonl", options.launch.logging.request_jsonl.string()},
             {"default_output_tokens", options.launch.generation.max_tokens},
             {"default_thinking", options.launch.generation.enable_thinking
                                      ? Json(*options.launch.generation.enable_thinking)
                                      : Json(nullptr)},
             {"default_thinking_budget", std::move(default_thinking_budget)},
             {"default_preserve_thinking", options.launch.generation.preserve_thinking
                                               ? Json(*options.launch.generation.preserve_thinking)
                                               : Json(nullptr)}};
    record["artifact"] = Json{{"path", options.launch.engine.artifact_path.string()},
                              {"size_bytes", std::move(artifact_size)},
                              {"architecture", load.architecture},
                              {"name", load.model_name},
                              {"formats", load.weight_formats},
                              {"prefill_signature", load.prefill_signature},
                              {"bytes_read", load.artifact_bytes_read},
                              {"host_to_device_bytes", load.host_to_device_bytes},
                              {"max_rank_staging_bytes", load.max_rank_staging_bytes},
                              {"device_object_count", load.device_object_count},
                              {"host_object_count", load.host_object_count},
                              {"load_seconds", load.load_seconds},
                              {"upload_seconds", load.upload_seconds}};
    const ninfer::ContextCacheOptions& cache       = engine_options.context_cache;
    const ninfer::ContextCostSummary& context_cost = load.context_cost;
    record["engine"]                               = Json{
                                      {"device", engine_options.device},
                                      {"max_context", engine_options.max_context},
                                      {"context_capacity_mode", context_capacity_mode_name(engine_options.context_capacity.mode)},
                                      {"context_pools", product::context_capacity_json(description.context_pools)},
                                      {"context_token_capacity_estimate", description.context_token_capacity_estimate},
                                      {"max_concurrency", engine_options.max_concurrency},
                                      {"max_pending_requests", engine_options.max_pending_requests},
                                      {"pending_timeout_ms", engine_options.pending_timeout_ms},
                                      {"prefill_chunk", engine_options.prefill_chunk},
                                      {"log_stats_interval_ms", options.launch.logging.stats_interval_ms},
                                      {"kv_cache", kv_cache_name(engine_options.kv_cache)},
                                      {"vision", engine_options.enable_vision},
                                      {"cuda_graph", engine_options.use_cuda_graph},
                                      {"prefix_reuse", options.launch.engine.context_cache.enabled},
                                      {"speculative_backend",
                                       product::speculative_backend_name(engine_options.speculative.backend)},
                                      {"speculative_draft_window", engine_options.speculative.draft_tokens},
                                      {"proposal_head", proposal_head_name(engine_options.speculative.proposal_head)},
                                      {"context_cost", Json{{"transfer_source", ninfer::context_cost_preset_source_name(
                                                      context_cost.transfer_source)},
                                                            {"prefill_source", ninfer::context_cost_preset_source_name(
                                                     context_cost.prefill_source)},
                                                            {"hardware_class", context_cost.hardware_class},
                                                            {"prefill_signature", context_cost.prefill_signature},
                                                            {"preset_path", context_cost.preset_path.string()}}},
                                      {"context_cache", Json{{"enabled", cache.enabled},
                                                             {"host_capacity_bytes", cache.host_capacity_bytes.value()}}}};
    record["sampling_defaults"] =
        Json{{"thinking", preset_json(sampling_defaults.thinking)},
             {"non_thinking", preset_json(sampling_defaults.non_thinking)},
             {"server_overrides", overrides_json(options.launch.generation.sampling)},
             {"omitted_seed", "random"}};
    record["resources"] = product::startup_resources_json(description, startup, snapshot);
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
    record["argv"] = options.launch.provenance.argv;
    return record.dump();
}

std::string
format_request_scheduling_json(const std::string& server_instance_id,
                               std::uint64_t timestamp_unix_ms, std::uint64_t request_id,
                               const std::string& http_request_id,
                               const ninfer::GenerationSchedulingObservation& observation) {
    Json record       = event_base(server_instance_id, timestamp_unix_ms, "request_scheduling");
    record["request"] = Json{{"request_id", request_id}, {"http_request_id", http_request_id}};
    record["engine_request_id"] = observation.engine_request_id;
    record["transition"]        = scheduling_transition_name(observation.transition);
    record["route"]             = recovery_route_json(observation.route);
    record["steady_ns"]         = observation.steady_ns;
    record["elapsed_ns"]        = observation.elapsed_ns;
    record["preemption_index"]  = observation.preemption_index;
    record["progress"]          = Json{
                 {"global_prefill_tokens", observation.global_prefill_tokens},
                 {"global_decode_tokens", observation.global_decode_tokens},
                 {"global_replayed_tokens", observation.global_replayed_tokens},
                 {"request_prefill_tokens", observation.request_prefill_tokens},
                 {"request_decode_tokens", observation.request_decode_tokens},
                 {"request_replayed_tokens", observation.request_replayed_tokens},
    };
    return record.dump();
}

std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp, const RequestLogContext& context) {
    Json record                   = event_base(server_instance_id, timestamp, "request_start");
    record["request"]             = request_json(context);
    record["preparation_seconds"] = preparation_json(context);
    return record.dump();
}

std::string format_request_rejected_json(const std::string& server_instance_id,
                                         std::uint64_t timestamp,
                                         const RequestRejectionLogContext& context) {
    Json record       = event_base(server_instance_id, timestamp, "request_rejected");
    record["phase"]   = "prepare";
    record["request"] = rejected_request_json(context);
    record["error"]   = error_json(context.error);
    return record.dump();
}

std::string format_request_done_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    Json record       = event_base(server_instance_id, timestamp, "request_done");
    record["request"] = request_json(context);
    record["result"]  = Json{
         {"finish_reason", finish_reason_name(outcome.finish_reason)},
         {"prompt_tokens", outcome.prompt_tokens},
         {"completion_tokens", outcome.completion_tokens},
         {"generated_token_ids", outcome.generated_token_ids},
         {"computed_prefill_tokens", outcome.stats.computed_prefill_tokens},
         {"prefix_cache_hit_tokens", outcome.stats.reused_prompt_tokens},
         {"prefix_reuse_path", prefix_reuse_path_name(outcome.stats.prefix_reuse_path)},
         {"thinking_budget", outcome.stats.thinking.configured_budget
                                 ? Json(*outcome.stats.thinking.configured_budget)
                                 : Json(nullptr)},
         {"model_thinking_tokens", outcome.stats.thinking.model_thinking_tokens},
         {"thinking_control_tokens", outcome.stats.thinking.injected_tokens},
         {"thinking_control_applied", outcome.stats.thinking.applied},
         {"tool_call_count",
          std::count_if(outcome.tool_calls.begin(), outcome.tool_calls.end(),
                        [](const auto& call) { return call.status == ToolCallStatus::Complete; })},
         {"tool_call_parse", tool_call_parse_json(outcome.tool_call_parse)}};
    record["timings_seconds"] =
        Json{{"prepare", outcome.service_prepare_seconds + outcome.stats.timings.prepare_seconds},
             {"ttft", outcome.stats.timings.first_token_seconds + outcome.service_prepare_seconds},
             {"vision_stream", outcome.stats.timings.vision_seconds},
             {"prefill", outcome.stats.timings.prefill_seconds},
             {"decode", outcome.stats.timings.decode_seconds},
             {"total", outcome.service_prepare_seconds + outcome.stats.timings.total_seconds}};
    record["execution"]               = product::request_execution_json(outcome.stats);
    record["service_prepare_seconds"] = outcome.service_prepare_seconds;
    record["engine_timing"]           = request_engine_timing_json(outcome.stats.engine_timing);
    record["constraint"] = product::constraint_observation_json(outcome.stats.constraint);
    record["first_output_timing"] =
        outcome.stats.first_output_timing
            ? first_output_timing_json(*outcome.stats.first_output_timing)
            : Json(nullptr);
    record["speculative"] = speculative_json(outcome.stats);
    record["generation"]  = Json{{"engine_request_id", outcome.stats.engine_request_id},
                                 {"scheduling", scheduling_json(outcome.stats.scheduling)},
                                 {"admission", admission_json(outcome.stats.admission)}};
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

std::string format_throughput_json(const std::string& server_instance_id, std::uint64_t timestamp,
                                   const ThroughputReport& report) {
    Json record                          = event_base(server_instance_id, timestamp, "throughput");
    const ninfer::RuntimeStats& previous = report.previous;
    const ninfer::RuntimeStats& current  = report.current;
    const double prefill_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.computed_prefill_tokens) / report.interval_seconds
            : 0.0;
    const double decode_rate =
        report.interval_seconds > 0.0
            ? static_cast<double>(report.committed_decode_tokens) / report.interval_seconds
            : 0.0;
    Json average_batch = nullptr;
    if (report.decode_rounds != 0) {
        average_batch = static_cast<double>(report.decode_row_rounds) /
                        static_cast<double>(report.decode_rounds);
    }
    const ninfer::RuntimeHostWorkStats host =
        host_work_delta(previous.host_work, current.host_work);
    const std::uint64_t active_host = host_active_ns(host);
    const auto window               = [&](ExecutionKind kind) {
        const auto i = static_cast<std::size_t>(kind);
        const auto value =
            product::execution_window(report.previous_execution[i], report.current_execution[i]);
        return Json{
                          {"units", value.units},
                          {"rows", value.rows},
                          {"input_tokens", value.input_tokens},
                          {"scheduled_positions", value.scheduled_positions},
                          {"elapsed_seconds", value.elapsed_seconds},
                          {"mean_seconds", value.mean_seconds() ? Json(*value.mean_seconds()) : Json(nullptr)}};
    };
    record["execution"] = {{"prefill", window(ExecutionKind::Prefill)},
                           {"decode", window(ExecutionKind::Decode)},
                           {"replay", window(ExecutionKind::Replay)},
                           {"control", window(ExecutionKind::Control)}};
    const auto drafts   = current.speculative_draft_tokens - previous.speculative_draft_tokens;
    const auto accepted =
        current.speculative_accepted_tokens - previous.speculative_accepted_tokens;
    record["speculative"] = {
        {"drafted_tokens", drafts},
        {"accepted_tokens", accepted},
        {"rounds", current.speculative_rounds - previous.speculative_rounds},
        {"fallback_steps",
         current.speculative_fallback_steps - previous.speculative_fallback_steps},
        {"acceptance", drafts ? Json(double(accepted) / drafts) : Json(nullptr)}};
    record["interval_seconds"] = report.interval_seconds;
    record["final_interval"]   = report.final_interval;
    record["tokens"]           = Json{{"computed_prefill", report.computed_prefill_tokens},
                                      {"committed_decode", report.committed_decode_tokens}};
    record["throughput_tokens_per_second"] =
        Json{{"prefill", prefill_rate}, {"decode", decode_rate}};
    record["scheduler"]  = Json{{"running", current.running_requests},
                                {"prefilling", current.prefilling_requests},
                                {"decode_ready", current.decode_ready_requests},
                                {"waiting", current.waiting_requests},
                                {"paused", current.paused_requests},
                                {"replaying", current.replaying_requests},
                                {"materializing", current.materializing_requests},
                                {"capture_pending", current.capture_pending_requests},
                                {"terminal_pending", current.terminal_pending_requests}};
    record["scheduling"] = Json{
        {"preemptions", monotonic_delta(previous.preemptions, current.preemptions)},
        {"snapshot_restores",
         monotonic_delta(previous.snapshot_restores, current.snapshot_restores)},
        {"replay_restores", monotonic_delta(previous.replay_restores, current.replay_restores)},
        {"replayed_tokens", monotonic_delta(previous.replayed_tokens, current.replayed_tokens)}};
    record["decode_batch"] = Json{{"rounds", report.decode_rounds},
                                  {"row_rounds", report.decode_row_rounds},
                                  {"average_size", std::move(average_batch)}};
    record["host_work"]    = Json{
           {"elapsed_seconds",
            Json{{"engine_boundary", nanoseconds_to_seconds(host.engine_boundary_ns)},
                 {"session_call", nanoseconds_to_seconds(host.session_call_ns)},
                 {"executor_poll", nanoseconds_to_seconds(host.executor_poll_ns)},
                 {"executor_flush", nanoseconds_to_seconds(host.executor_flush_ns)},
                 {"engine_commit_output", nanoseconds_to_seconds(host.engine_commit_output_ns)},
                 {"engine_maintenance", nanoseconds_to_seconds(host.engine_maintenance_ns)},
                 {"total", nanoseconds_to_seconds(active_host)}}},
           {"scheduler_yield_seconds", nanoseconds_to_seconds(host.scheduler_yield_ns)},
           {"executor_flush_seconds", nanoseconds_to_seconds(host.executor_flush_ns)},
           {"work_class_seconds",
            Json{
                {"decode_host", nanoseconds_to_seconds(host.decode_host_ns)},
                {"decode_scheduler_yield", nanoseconds_to_seconds(host.decode_scheduler_yield_ns)},
                {"prefill_host", nanoseconds_to_seconds(host.prefill_host_ns)},
                {"prefill_scheduler_yield", nanoseconds_to_seconds(host.prefill_scheduler_yield_ns)},
                {"control_host", nanoseconds_to_seconds(host.control_host_ns)},
                {"control_scheduler_yield", nanoseconds_to_seconds(host.control_scheduler_yield_ns)}}},
           {"detail_subset_seconds",
            Json{{"stats_publication", nanoseconds_to_seconds(host.stats_publication_ns)}}},
           {"detail_invocations", Json{{"stats_publication", host.stats_publication_invocations}}},
           {"units", Json{{"prefill", host.prefill_units}, {"control", host.control_units}}},
           {"decode_host_microseconds_per_round",
            microseconds_per(host.decode_host_ns, report.decode_rounds)},
           {"decode_host_microseconds_per_row_round",
            microseconds_per(host.decode_host_ns, report.decode_row_rounds)},
           {"decode_scheduler_yield_microseconds_per_round",
            microseconds_per(host.decode_scheduler_yield_ns, report.decode_rounds)},
           {"detail_microseconds_per_invocation",
            Json{{"stats_publication",
                  microseconds_per(host.stats_publication_ns, host.stats_publication_invocations)}}},
    };
    record["context_cache"] = Json{
        {"captures", Json{{"completed", monotonic_delta(previous.active_captures_completed,
                                                        current.active_captures_completed)},
                          {"aborted", monotonic_delta(previous.active_captures_aborted,
                                                      current.active_captures_aborted)}}},
        {"selections",
         Json{{"root", monotonic_delta(previous.root_selections, current.root_selections)},
              {"checkpoint",
               monotonic_delta(previous.checkpoint_selections, current.checkpoint_selections)},
              {"reused_prompt_tokens",
               monotonic_delta(previous.reused_prompt_tokens, current.reused_prompt_tokens)}}},
        {"last_selection", Json{{"frontier_tokens", current.last_selected_frontier_tokens}}},
        {"state_operations",
         Json{{"moves", monotonic_delta(previous.state_moves, current.state_moves)},
              {"forks", monotonic_delta(previous.state_forks, current.state_forks)},
              {"materialization_forks", monotonic_delta(previous.materialization_state_forks,
                                                        current.materialization_state_forks)},
              {"restores", monotonic_delta(previous.state_restores, current.state_restores)}}},
        {"state_transfers",
         transfer_timings_json(current.context_transfers.state, previous.context_transfers.state)},
        {"pressure",
         Json{{"spill_pages",
               monotonic_delta(previous.pressure_spill_pages, current.pressure_spill_pages)},
              {"partial_tail_cow_pages",
               monotonic_delta(previous.partial_tail_cow_pages, current.partial_tail_cow_pages)}}},
        {"occupancy", product::context_usage_json(current)},
        {"actual_transfer_seconds", monotonic_delta(previous.actual_context_transfer_seconds,
                                                    current.actual_context_transfer_seconds)}};
    Json pools = Json::array();
    for (const auto& pool : current.kv_pools) {
        const auto before = pool.id < previous.context_transfers.kv.size()
                                ? previous.context_transfers.kv[pool.id]
                                : ninfer::ContextTransferSeries{};
        pools.push_back({{"pool_id", pool.id},
                         {"label", pool.label},
                         {"occupied_page_groups", pool.occupied_page_groups},
                         {"transfers", transfer_timings_json(
                                           current.context_transfers.kv.at(pool.id), before)}});
    }
    record["context_cache"]["kv_pools"] = std::move(pools);
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
                                 const std::string& protected_artifact_path,
                                 std::shared_ptr<spdlog::logger> logger)
    : path_(path), logger_(std::move(logger)) {
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

void JsonlRequestLog::write_startup_config(const product::ServeOptions& options) {
    if (enabled()) append(format_startup_config_json(server_instance_id_, unix_time_ms(), options));
}

void JsonlRequestLog::write_server_start(const product::ServeOptions& options,
                                         const ninfer::EngineOptions& engine_options,
                                         const ninfer::ModelSamplingDefaults& sampling_defaults,
                                         const std::string& public_model_id,
                                         const EngineDescription& description,
                                         const StartupReport& startup,
                                         const EngineSnapshot& snapshot) {
    if (!enabled()) { return; }
    std::error_code error;
    const std::uintmax_t size =
        std::filesystem::file_size(options.launch.engine.artifact_path, error);
    const std::optional<std::uint64_t> artifact_size =
        error ? std::nullopt : std::optional<std::uint64_t>(size);
    append(format_server_start_json(
        server_instance_id_, unix_time_ms(), options, engine_options, sampling_defaults,
        public_model_id, description, startup, snapshot,
        query_server_log_environment(options.launch.engine.device), artifact_size));
}

void JsonlRequestLog::write_request_start(const RequestLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_start_json(server_instance_id_, unix_time_ms(), context));
}

void JsonlRequestLog::write_request_scheduling(
    std::uint64_t request_id, const std::string& http_request_id,
    const ninfer::GenerationSchedulingObservation& observation) {
    if (!enabled()) { return; }
    append(format_request_scheduling_json(server_instance_id_, unix_time_ms(), request_id,
                                          http_request_id, observation));
}

void JsonlRequestLog::write_request_rejected(const RequestRejectionLogContext& context) {
    if (!enabled()) { return; }
    append(format_request_rejected_json(server_instance_id_, unix_time_ms(), context));
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

void JsonlRequestLog::write_throughput(const ThroughputReport& report) {
    if (!enabled()) { return; }
    append(format_throughput_json(server_instance_id_, unix_time_ms(), report));
}

void JsonlRequestLog::append(std::string record) {
    bool report_failure = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (failed_) { return; }
        output_ << record << '\n';
        output_.flush();
        if (!output_) {
            failed_        = true;
            report_failure = true;
        }
    }
    if (report_failure && logger_ != nullptr) {
        logger_->error("request log disabled | write failed | {}",
                       product::format_pretty_text(path_));
    }
}

} // namespace ninfer::serve
