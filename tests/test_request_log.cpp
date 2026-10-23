#include "serve/operational_log.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using namespace ninfer::serve;
using ninfer::product::ServeOptions;
using Json = nlohmann::json;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    bool protected_artifact_rejected = false;
    try {
        JsonlRequestLog unsafe("same-path.ninfer", "same-path.ninfer");
    } catch (const std::invalid_argument&) { protected_artifact_rejected = true; }
    failures += check(protected_artifact_rejected,
                      "request log accepted the model artifact as its output path");

    ServeOptions options;
    options.launch.engine.artifact_path  = "/models/qwen3_6_27b.ninfer";
    options.host                         = "127.0.0.1";
    options.port                         = 8123;
    options.api_key                      = "must-not-appear";
    options.model_id_override            = "deployment-alias";
    options.launch.logging.request_jsonl = "requests.jsonl";
    options.launch.engine.max_context    = 262144;
    options.launch.engine.context_capacity =
        ninfer::ContextCapacityPolicy::explicit_capacity(8ULL << 30);
    options.launch.engine.prefill_chunk             = 1024;
    options.launch.logging.stats_interval_ms        = 2500;
    options.launch.engine.kv_cache                  = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.launch.engine.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.launch.engine.speculative.draft_tokens  = 3;
    options.launch.engine.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.launch.engine.enable_vision             = false;
    options.launch.engine.context_cache.enabled     = true;
    options.launch.generation.preserve_thinking     = true;
    options.launch.generation.thinking_budget       = 512;
    options.launch.generation.sampling.temperature  = 0.6F;
    options.launch.provenance.argv = {"ninfer-serve", options.launch.engine.artifact_path.string(),
                                      "--api-key", "<redacted>"};

    const Json startup = Json::parse(format_startup_config_json("test-startup", 1000, options));
    failures += check(startup.at("event") == "startup_config" &&
                          startup.at("schema_version") == kRequestLogSchemaVersion &&
                          startup.at("configuration").at("generation.sampling.temperature") ==
                              static_cast<double>(0.6F) &&
                          startup.at("configuration").at("server.api_key") == "configured" &&
                          startup.dump().find("must-not-appear") == std::string::npos,
                      "startup configuration log lost exact values or exposed the API key");

    ninfer::EngineOptions engine_options;
    engine_options.artifact_path                     = options.launch.engine.artifact_path;
    engine_options.device                            = options.launch.engine.device;
    engine_options.max_context                       = options.launch.engine.max_context;
    engine_options.max_concurrency                   = 2;
    engine_options.max_pending_requests              = options.launch.engine.max_pending_requests;
    engine_options.pending_timeout_ms                = options.launch.engine.pending_timeout_ms;
    engine_options.prefill_chunk                     = options.launch.engine.prefill_chunk;
    engine_options.kv_cache                          = options.launch.engine.kv_cache;
    engine_options.speculative                       = options.launch.engine.speculative;
    engine_options.enable_vision                     = options.launch.engine.enable_vision;
    engine_options.use_cuda_graph                    = options.launch.engine.use_cuda_graph;
    engine_options.context_capacity                  = options.launch.engine.context_capacity;
    engine_options.context_cache.host_capacity_bytes = 64ULL << 20;

    const ninfer::ModelSamplingDefaults sampling_defaults{
        .thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F},
        .non_thinking = {.temperature = 0.7F, .top_k = 20, .top_p = 0.8F, .presence_penalty = 1.5F},
    };

    ninfer::LoadSummary load;
    load.architecture           = "Qwen3_5ForCausalLM";
    load.model_name             = "qwen3.6-27b";
    load.weight_formats         = {"q4_g64_fp16", "q8_g32_fp16"};
    load.load_seconds           = 1.234567890123;
    load.upload_seconds         = 0.345678901234;
    load.artifact_bytes_read    = 1000;
    load.host_to_device_bytes   = 900;
    load.max_rank_staging_bytes = 128;
    load.device_object_count    = 42;
    load.host_object_count      = 6;
    load.context_cost           = {
                  .transfer_source   = ninfer::ContextCostPresetSource::External,
                  .prefill_source    = ninfer::ContextCostPresetSource::CompiledDefault,
                  .hardware_class    = "nvidia-geforce-rtx-5090-sm120",
                  .prefill_signature = "example-prefill-signature",
                  .preset_path       = "local-costs.json",
    };

    ninfer::EngineDescription description;
    description.ranks.push_back({});
    description.context_pools = {{0, ninfer::ContextMemoryDomain::Device, 8ULL << 30, 153954304},
                                 {0, ninfer::ContextMemoryDomain::Host, 64ULL << 20, 153954304}};
    description.kv_pools.push_back({0, "main", 0});
    ninfer::StartupReport startup_report;
    startup_report.load = load;
    startup_report.ranks.resize(1);
    ninfer::EngineSnapshot snapshot;
    snapshot.ranks.resize(1);
    auto& memory                               = snapshot.ranks[0].memory;
    auto& budget                               = startup_report.ranks[0];
    description.max_context                    = 262144;
    description.kv_cache                       = ninfer::KvCacheStorage::Fp8E4M3Row256;
    memory.allocations.device_weights_bytes    = 100;
    memory.allocations.persistent_device_bytes = 200;
    memory.allocations.workspace_device_bytes  = 500;
    memory.vision                              = ninfer::VisionWorkspaceMemorySummary{
                                     .aggregate_prompt_tokens = 32768,
                                     .max_item_tokens         = 16384,
                                     .general_capacity_bytes  = 300,
                                     .encode_required_bytes   = 400,
                                     .handoff_offset_bytes    = 300,
                                     .handoff_capacity_bytes  = 200,
                                     .handoff_active_bytes    = 0,
                                     .handoff_peak_bytes      = 150,
    };
    budget.minimum_runtime_reservation_bytes = 1300;
    budget.context_capacity_bytes            = 8ULL << 30;
    budget.runtime_reservation_bytes         = 1600;
    budget.available_after_weights_bytes     = 1700;
    budget.available_after_startup_bytes     = 180;
    budget.planned_slack_bytes               = 100;
    budget.graphs                            = ninfer::GraphMemorySummary{600, -30, 800, 200};
    budget.retained_bytes                    = 1000;
    snapshot.runtime.context_pools           = {{.rank           = 0,
                                                 .domain         = ninfer::ContextMemoryDomain::Host,
                                                 .capacity_bytes = 64ULL << 20,
                                                 .live_bytes     = 8ULL << 20,
                                                 .reserved_bytes = 4ULL << 20}};

    ServerLogEnvironment environment;
    environment.device                    = 0;
    environment.gpu_name                  = "NVIDIA GeForce RTX 5090";
    environment.gpu_uuid                  = "GPU-00000000-0000-0000-0000-000000000000";
    environment.total_device_memory_bytes = 32000000000ULL;
    environment.compute_capability_major  = 12;
    environment.compute_capability_minor  = 0;
    environment.cuda_compile_version      = "13.1";
    environment.cuda_runtime_version      = "13.1";
    environment.cuda_driver_version       = "13.1";

    const Json server = Json::parse(format_server_start_json(
        "serve-test", 1000, options, engine_options, sampling_defaults, "deployment-alias",
        description, startup_report, snapshot, environment, std::uint64_t{123456}));
    failures += check(server.at("artifact_type") == kRequestLogArtifactType,
                      "server record artifact type mismatch");
    failures += check(server.at("schema_version") == kRequestLogSchemaVersion,
                      "server record schema mismatch");
    failures += check(server.at("event") == "server_start", "server event mismatch");
    failures += check(server.at("server").at("public_model_id") == "deployment-alias",
                      "resolved public model id missing");
    failures += check(server.at("artifact").at("architecture") == "Qwen3_5ForCausalLM",
                      "server target missing");
    failures +=
        check(server.at("artifact").at("formats") == Json::array({"q4_g64_fp16", "q8_g32_fp16"}),
              "server weights id missing");
    failures += check(server.at("artifact").at("size_bytes") == 123456, "artifact size missing");
    failures += check(server.at("engine").at("max_context") == 262144, "max context missing");
    failures += check(server.at("engine").at("context_capacity_mode") == "explicit" &&
                          server.at("engine").at("context_pools").at(0).at("capacity_bytes") ==
                              (8ULL << 30) &&
                          server.at("engine").at("context_pools").at(1).at("domain") == "host",
                      "rank-local Context capacity metadata missing");
    failures +=
        check(server.at("engine").at("log_stats_interval_ms") == 2500, "stats interval missing");
    failures += check(server.at("server").at("request_log_jsonl") == "requests.jsonl",
                      "request log path missing");
    failures += check(server.at("server").at("default_thinking_budget") == 512,
                      "server thinking budget missing");
    failures += check(server.at("engine").at("kv_cache") == "fp8-e4m3-row256", "KV type missing");
    options.launch.engine.kv_cache = ninfer::KvCacheStorage::Nvfp4Group16;
    engine_options.kv_cache        = options.launch.engine.kv_cache;
    description.kv_cache           = options.launch.engine.kv_cache;
    const Json nvfp4_server        = Json::parse(format_server_start_json(
        "serve-test", 1000, options, engine_options, sampling_defaults, "deployment-alias",
        description, startup_report, snapshot, environment, std::uint64_t{123456}));
    failures +=
        check(nvfp4_server.at("engine").at("kv_cache") == "nvfp4", "NVFP4 KV report name missing");
    options.launch.engine.kv_cache = ninfer::KvCacheStorage::Fp8KeyNvfp4Value;
    engine_options.kv_cache        = options.launch.engine.kv_cache;
    description.kv_cache           = options.launch.engine.kv_cache;
    const Json k8v4_server         = Json::parse(format_server_start_json(
        "serve-test", 1000, options, engine_options, sampling_defaults, "deployment-alias",
        description, startup_report, snapshot, environment, std::uint64_t{123456}));
    failures +=
        check(k8v4_server.at("engine").at("kv_cache") == "k8v4", "K8V4 KV report name missing");
    failures += check(server.at("engine").at("vision") == false, "Vision state missing");
    failures += check(server.at("engine").at("speculative_backend") == "mtp",
                      "speculative backend missing");
    failures +=
        check(server.at("engine").at("proposal_head") == "optimized", "proposal head missing");
    failures += check(
        server.at("engine").at("context_cost").at("transfer_source") == "external" &&
            server.at("engine").at("context_cost").at("prefill_source") == "compiled-default" &&
            server.at("engine").at("context_cost").at("hardware_class") ==
                "nvidia-geforce-rtx-5090-sm120" &&
            server.at("engine").at("context_cost").at("preset_path") == "local-costs.json",
        "resolved context-cost layers missing");
    failures += check(server.at("engine").at("prefix_reuse") == true, "prefix-reuse state missing");
    failures +=
        check(server.at("engine").at("context_cache").at("host_capacity_bytes") == (64ULL << 20),
              "resolved context-cache configuration missing");
    failures += check(server.at("server").at("default_preserve_thinking") == true,
                      "server preserve-thinking default missing");
    failures +=
        check(server.at("sampling_defaults").at("thinking").at("temperature") == 1.0 &&
                  server.at("sampling_defaults").at("non_thinking").at("presence_penalty") == 1.5,
              "registered mode-specific sampling defaults missing");
    failures += check(
        server.at("sampling_defaults").at("server_overrides").at("temperature").get<float>() ==
                0.6F &&
            server.at("sampling_defaults").at("server_overrides").at("top_p").is_null(),
        "server sampling overrides lost omission state");
    failures += check(server.at("environment").at("gpu_name") == "NVIDIA GeForce RTX 5090",
                      "GPU name missing");
    failures += check(server.at("resources")
                                  .at("ranks")
                                  .at(0)
                                  .at("resources")
                                  .at("allocations")
                                  .at("workspace_device_bytes") == 500 &&
                          server.at("resources")
                                  .at("ranks")
                                  .at(0)
                                  .at("resources")
                                  .at("vision_workspace")
                                  .at("general_capacity_bytes") == 300 &&
                          server.at("resources")
                                  .at("ranks")
                                  .at(0)
                                  .at("resources")
                                  .at("vision_workspace")
                                  .at("handoff_capacity_bytes") == 200 &&
                          server.at("resources")
                                  .at("ranks")
                                  .at(0)
                                  .at("resources")
                                  .at("vision_workspace")
                                  .at("handoff_peak_bytes") == 150,
                      "Vision workspace layout missing");
    failures += check(server.at("resources")
                              .at("ranks")
                              .at(0)
                              .at("budget")
                              .at("graph_calibration")
                              .at("rebuild_estimate_bytes") == 600,
                      "CUDA Graph rebuild estimate missing");
    failures += check(
        server.at("resources")
                    .at("ranks")
                    .at(0)
                    .at("budget")
                    .at("graph_calibration")
                    .at("preparation_increment_bytes") == -30 &&
            server.at("resources")
                    .at("ranks")
                    .at(0)
                    .at("budget")
                    .at("graph_calibration")
                    .at("calibration_increment_bytes") == 800 &&
            server.at("resources")
                    .at("ranks")
                    .at(0)
                    .at("budget")
                    .at("graph_calibration")
                    .at("calibration_retained_bytes") == 200 &&
            server.at("resources").at("ranks").at(0).at("budget").at("retained_bytes") == 1000,
        "signed Graph and startup deltas missing");
    failures += check(
        server.at("resources").at("ranks").at(0).at("budget").at("runtime_reservation_bytes") ==
                1600 &&
            server.at("resources")
                    .at("ranks")
                    .at(0)
                    .at("budget")
                    .at("available_after_weights_bytes") == 1700 &&
            server.at("resources")
                    .at("ranks")
                    .at(0)
                    .at("budget")
                    .at("available_after_startup_bytes") == 180 &&
            server.at("resources").at("ranks").at(0).at("budget").at("headroom_bytes") == 0 &&
            server.at("resources").at("ranks").at(0).at("budget").at("planned_slack_bytes") == 100,
        "adaptive KV memory ledger missing");
    failures +=
        check(server.at("engine").at("context_cache").at("host_capacity_bytes") == (64ULL << 20),
              "Host context capacity missing");
    failures += check(server.dump().find("must-not-appear") == std::string::npos,
                      "server JSON leaked the API key");
    failures += check(server.at("argv").at(3) == "<redacted>",
                      "server argv did not retain the redaction marker");

    GenerationRequest request;
    request.max_tokens = 4096;
    request.messages.resize(2);
    request.messages.front().content.push_back(ContentPart{.kind = ContentKind::Image});

    PreparedRequest prepared;
    prepared.enable_thinking                           = true;
    prepared.thinking_budget                           = 256;
    prepared.reasoning_effort                          = ninfer::ReasoningEffort::XHigh;
    prepared.preserve_thinking                         = true;
    prepared.sampling.temperature                      = 0.6F;
    prepared.sampling.top_p                            = 0.95F;
    prepared.sampling.top_k                            = 20;
    prepared.sampling.min_p                            = 0.0F;
    prepared.sampling.presence_penalty                 = 1.0F;
    prepared.sampling.frequency_penalty                = 0.0F;
    prepared.sampling.seed                             = 7632647173703958409ULL;
    prepared.acquisition_seconds                       = 0.004;
    prepared.preparation.seconds                       = 0.12;
    prepared.preparation.media_preprocess_seconds      = 0.08;
    prepared.preparation.media_preprocess_work_seconds = 0.31;
    prepared.preparation.tokenize_seconds              = 0.02;
    prepared.preparation.media_items                   = 1;
    prepared.preparation.media_cache_misses            = 1;
    prepared.preparation.built_patch_bytes             = 49152;

    const RequestLogMetadata metadata{
        .http_request_id                   = "req_wire_923",
        .response_id                       = "chatcmpl_wire_471",
        .model                             = "qwen3.6-27b",
        .stream                            = false,
        .output_tokens_explicit            = true,
        .preserve_thinking_semantic_change = true,
    };
    const RequestLogContext context =
        make_request_log_context(7, "openai_chat_completions", request, metadata, prepared);
    const OperationalRecord pretty_start = render_request_start(context);
    failures += check(
        pretty_start.message ==
            "req#7 started | openai-chat non-stream | 2 messages | max output 4,096 | thinking "
            "xhigh, budget 256 | media 1, prepared 120 ms | preserve thinking",
        "pretty request-start record mismatch");
    RequestLogContext default_thinking = context;
    default_thinking.requested_reasoning_effort.reset();
    default_thinking.thinking_budget.reset();
    const std::string default_thinking_start = render_request_start(default_thinking).message;
    failures +=
        check(default_thinking_start.find("thinking template default") != std::string::npos &&
                  default_thinking_start.find("unresolved") == std::string::npos,
              "default thinking state leaks an internal resolution detail");
    const Json started = Json::parse(format_request_start_json("serve-test", 2000, context));
    failures +=
        check(started.at("request").at("request_id") == 7, "request id missing from start record");
    failures += check(started.at("request").at("http_request_id") == "req_wire_923" &&
                          started.at("request").at("response_id") == "chatcmpl_wire_471",
                      "public wire identities were not preserved in the start record");
    failures += check(started.at("request").at("requested_output_tokens") == 4096,
                      "request output budget missing");
    failures += check(started.at("request").at("enable_thinking") == true,
                      "resolved thinking mode missing");
    failures += check(started.at("request").at("thinking_budget") == 256,
                      "resolved thinking budget missing");
    failures += check(started.at("request").at("requested_reasoning_effort") == "xhigh" &&
                          !started.at("request").contains("resolved_reasoning_effort"),
                      "requested and resolved reasoning effort are not distinguished");
    failures += check(started.at("request").at("preserve_thinking") == true &&
                          started.at("request").at("preserve_thinking_semantic_change") == true,
                      "resolved preserve-thinking metadata missing");
    failures += check(started.at("request").at("sampling").at("seed") == 7632647173703958409ULL,
                      "resolved seed missing");
    failures += check(started.at("request").at("media_item_count") == 1 &&
                          started.at("preparation_seconds").at("acquisition") == 0.004 &&
                          started.at("preparation_seconds").at("media_preprocess_work") == 0.31 &&
                          started.at("preparation_seconds").at("tokenize") == 0.02 &&
                          started.at("preparation_seconds").at("cache_misses") == 1,
                      "request-scoped media preparation diagnostics missing");

    ApiError preparation_error;
    preparation_error.status             = 400;
    preparation_error.type               = "invalid_request_error";
    preparation_error.param              = "messages";
    preparation_error.code               = "context_length_exceeded";
    preparation_error.message            = "sentinel-client-value\nsecond-record";
    GenerationRequest rejected_request   = request;
    rejected_request.reasoning_effort    = RequestedReasoningEffort::High;
    RequestLogMetadata rejected_metadata = metadata;
    rejected_metadata.http_request_id    = "req_rejected_812";
    rejected_metadata.response_id.clear();
    const RequestRejectionLogContext rejected_context = make_request_rejection_log_context(
        8, "anthropic_messages", rejected_request, rejected_metadata, preparation_error);
    const Json rejected =
        Json::parse(format_request_rejected_json("serve-test", 2500, rejected_context));
    failures +=
        check(rejected.at("event") == "request_rejected" && rejected.at("phase") == "prepare",
              "preparation rejection event or phase mismatch");
    failures += check(rejected.at("request").at("request_id") == 8 &&
                          rejected.at("request").at("http_request_id") == "req_rejected_812" &&
                          rejected.at("request").at("response_id") == "" &&
                          rejected.at("request").at("media_item_count") == 1 &&
                          rejected.at("request").at("message_count") == 2,
                      "preparation rejection request shape missing");
    failures += check(rejected.at("request").at("requested_reasoning_effort") == "high" &&
                          !rejected.at("request").contains("resolved_reasoning_effort"),
                      "rejection log fabricated a resolved reasoning effort");
    failures += check(rejected.at("error").at("status") == 400 &&
                          rejected.at("error").at("code") == "context_length_exceeded" &&
                          rejected.at("error").at("param") == "messages" &&
                          rejected.at("error").at("message") == preparation_error.message,
                      "preparation rejection API error missing");
    const OperationalRecord client_rejection = render_request_rejected(rejected_context);
    failures += check(
        client_rejection.severity == OperationalSeverity::Info &&
            client_rejection.message.find("req#8 rejected during prepare") != std::string::npos &&
            client_rejection.message.find("context length exceeded") != std::string::npos &&
            client_rejection.message.find("sentinel-client-value") == std::string::npos &&
            client_rejection.message.find('\n') == std::string::npos,
        "operational rejection severity or client-data policy mismatch");
    RequestRejectionLogContext overload_context = rejected_context;
    overload_context.error.status               = 429;
    overload_context.error.code                 = "server_overloaded";
    failures +=
        check(render_request_rejected(overload_context).severity == OperationalSeverity::Warning,
              "operational overload rejection is not warning severity");

    GenerationOutcome outcome;
    outcome.prompt_tokens                         = 401;
    outcome.completion_tokens                     = 1024;
    outcome.finish_reason                         = ninfer::FinishReason::OutputLimit;
    outcome.stats.timings.prepare_seconds         = 0.1234567890123;
    outcome.stats.timings.first_token_seconds     = 0.3580246791357;
    outcome.stats.timings.vision_seconds          = 0.0;
    outcome.stats.timings.prefill_seconds         = 0.2345678901234;
    outcome.stats.timings.decode_seconds          = 5.3456789012345;
    outcome.stats.timings.generation_wall_seconds = 5.3456789012345;
    outcome.stats.timings.total_seconds           = 5.7037035803702;
    outcome.stats.reused_prompt_tokens            = 101;
    outcome.stats.computed_prefill_tokens         = 300;
    outcome.stats.prefix_reuse_path               = ninfer::PrefixReusePath::Checkpoint;
    outcome.stats.engine_timing                   = {
                          .queue_wait_seconds                     = 0.001,
                          .engine_boundary_exposed_seconds        = 0.001,
                          .session_call_exposed_seconds           = 0.002,
                          .executor_poll_exposed_seconds          = 0.003,
                          .engine_commit_output_exposed_seconds   = 0.004,
                          .engine_maintenance_exposed_seconds     = 0.005,
                          .scheduler_yield_exposed_seconds        = 0.3,
                          .decode_host_exposed_seconds            = 0.01,
                          .decode_scheduler_yield_exposed_seconds = 0.2,
                          .prefill_units                          = 4,
                          .decode_rounds                          = 2,
                          .control_units                          = 1,
    };
    outcome.stats.speculative.backend               = ninfer::SpeculativeBackend::Mtp;
    outcome.stats.speculative.draft_window          = 3;
    outcome.stats.speculative.rounds                = 300;
    outcome.stats.speculative.drafted_tokens        = 900;
    outcome.stats.speculative.accepted_tokens       = 720;
    outcome.stats.speculative.fallback_steps        = 2;
    outcome.stats.speculative.accepted_per_position = {290, 240, 190};
    outcome.stats.engine_request_id                 = 991;
    outcome.stats.scheduling                        = {
                               .preemptions          = 3,
                               .snapshot_restores    = 1,
                               .replay_restores      = 2,
                               .replayed_tokens      = 640,
                               .paused_ns            = 123456789012345ULL,
                               .device_to_host_bytes = 3145728,
                               .host_to_device_bytes = 2097152,
    };
    outcome.stats.thinking = ninfer::ThinkingBudgetStats{.configured_budget     = 256,
                                                         .model_thinking_tokens = 256,
                                                         .injected_tokens       = 19,
                                                         .applied               = true};

    const Json done = Json::parse(format_request_done_json("serve-test", 3000, context, outcome));
    failures += check(done.at("request").at("request_id") == 7 &&
                          done.at("request").at("http_request_id") == "req_wire_923" &&
                          done.at("request").at("response_id") == "chatcmpl_wire_471",
                      "terminal log cannot be joined to the same public request and response");
    failures +=
        check(done.at("generation").at("engine_request_id") == 991 &&
                  done.at("generation").at("scheduling") == Json{{"preemptions", 3},
                                                                 {"snapshot_restores", 1},
                                                                 {"replay_restores", 2},
                                                                 {"replayed_tokens", 640},
                                                                 {"paused_ns", 123456789012345ULL},
                                                                 {"device_to_host_bytes", 3145728},
                                                                 {"host_to_device_bytes", 2097152}},
              "per-request scheduling observations or Engine request ID missing");
    failures +=
        check(done.at("result").at("finish_reason") == "output_limit", "finish reason missing");
    failures += check(done.at("result").at("prompt_tokens") == 401, "prompt tokens missing");
    failures += check(done.at("result").at("computed_prefill_tokens") == 300,
                      "initial computed prompt tokens must exclude separately reported replay");
    auto cancelled                          = outcome;
    cancelled.finish_reason                 = ninfer::FinishReason::Cancelled;
    cancelled.completion_tokens             = 0;
    cancelled.stats.computed_prefill_tokens = 128;
    cancelled.stats.timings.prefill_seconds = 0.25;
    const Json cancelled_done =
        Json::parse(format_request_done_json("server-test", 123456789, context, cancelled));
    failures += check(cancelled_done.at("result").at("computed_prefill_tokens") == 128,
                      "cancelled prefill reported the uncomputed remainder as executed work");
    failures += check(cancelled_done.at("first_output_timing").is_null() &&
                          cancelled_done.at("result").at("generated_token_ids").empty(),
                      "cancelled before output fabricated an output boundary or generated tokens");
    failures += check(render_request_done(context, cancelled).message.find("prefill 512.0 tok/s") !=
                          std::string::npos,
                      "cancelled prefill rate must use completed work, not the remaining prompt");
    failures += check(done.at("result").at("prefix_reuse_path") == "checkpoint",
                      "prefix reuse path missing");
    failures += check(done.at("result").at("thinking_budget") == 256 &&
                          done.at("result").at("model_thinking_tokens") == 256 &&
                          done.at("result").at("thinking_control_tokens") == 19 &&
                          done.at("result").at("thinking_control_applied") == true,
                      "thinking-control result accounting missing");
    failures +=
        check(done.at("result").at("tool_call_parse").at("marker_seen") == false &&
                  done.at("result").at("tool_call_parse").at("structured_call_count") == 0 &&
                  done.at("result").at("tool_call_parse").at("schema_mismatch_arguments") == 0 &&
                  done.at("result").at("tool_call_parse").at("fallback_reason") == "none",
              "default tool-call parse diagnostics missing");
    failures += check(done.at("timings_seconds").at("decode").get<double>() ==
                          outcome.stats.timings.decode_seconds,
                      "decode time lost precision");
    failures += check(done.at("timings_seconds").at("ttft").get<double>() ==
                          outcome.stats.timings.first_token_seconds,
                      "TTFT missing or lost precision");
    failures += check(done.at("speculative").at("backend") == "mtp", "speculative backend missing");
    failures +=
        check(done.at("speculative").at("draft_window") == 3, "speculative draft window missing");
    failures += check(done.at("speculative").at("fallback_steps") == 2,
                      "speculative fallback count missing");
    failures +=
        check(done.at("speculative").at("accepted_per_position") == Json::array({290, 240, 190}),
              "speculative position counts missing");
    failures += check(
        done.at("engine_timing").at("queue_wait_seconds") == 0.001 &&
            std::abs(done.at("engine_timing").at("host_exposed_seconds").at("total").get<double>() -
                     0.015) < 1.0e-15 &&
            done.at("engine_timing").at("decode").at("rounds") == 2 &&
            done.at("engine_timing").at("units").at("prefill") == 4,
        "request Engine timing exposure is incomplete");

    auto observed                      = outcome;
    observed.generated_token_ids       = {17, 151645};
    observed.completion_tokens         = 2;
    observed.stats.first_output_timing = ninfer::GenerationFirstOutputTiming{
        .elapsed_seconds         = 0.312345678901234,
        .initial_binding_seconds = 0.023456789012345,
        .engine                  = {.queue_wait_seconds              = 0.001,
                                    .session_call_exposed_seconds    = 0.0015,
                                    .scheduler_yield_exposed_seconds = 0.12,
                                    .prefill_units                   = 3},
        .prefill    = {.result = {2, 2000000, 1200000}, .total = {2, 9000000, 5000000}},
        .replay     = {.result = {1, 6000000, 6000000}, .total = {1, 21000000, 21000000}},
        .scheduling = {.preemptions     = 1,
                       .replay_restores = 1,
                       .replayed_tokens = 64,
                       .paused_ns       = 123456789},
        .computed_prefill_tokens = 192,
    };
    auto& first = *observed.stats.first_output_timing;
    first.context_transfers.kv.resize(2);
    first.context_transfers.state[0] = {.bytes = 9007199254740993ULL, .seconds = 0.010};
    first.context_transfers.kv[0][1] = {.bytes = 2113536, .seconds = 0.011};
    first.context_transfers.kv[1][2] = {.bytes = 16384, .seconds = 0.012};
    const Json observed_done =
        Json::parse(format_request_done_json("serve-test", 3001, context, observed));
    const auto& first_json = observed_done.at("first_output_timing");
    failures += check(observed_done.at("schema_version") == kRequestLogSchemaVersion &&
                          observed_done.at("result").at("generated_token_ids") ==
                              Json::array({17, 151645}) &&
                          observed_done.at("result").at("completion_tokens") == 2,
                      "request diagnostics lost exact generated token IDs or schema version");
    observed.stats.admission = {.preferred_reused_tokens = 19057,
                                .source_wait_seconds     = 0.25,
                                .revoked_checkpoints     = 1,
                                .fallback_reason = ninfer::AdmissionFallbackReason::SourceRevoked};
    const auto admission_done =
        Json::parse(format_request_done_json("serve-test", 3002, context, observed));
    failures +=
        check(admission_done.at("generation").at("admission") ==
                  Json{{"preferred_reused_tokens", 19057},
                       {"source_wait_seconds", 0.25},
                       {"revoked_checkpoints", 1},
                       {"fallback_reason", "source_revoked"}},
              "admission observation lost selected source, queue subset or revocation reason");

    const ninfer::GenerationSchedulingObservation scheduling{
        .transition              = ninfer::GenerationSchedulingTransition::ReplayComplete,
        .route                   = ninfer::GenerationRecoveryRoute::Replay,
        .engine_request_id       = 19,
        .steady_ns               = 9007199254740993ULL,
        .elapsed_ns              = 123456789,
        .preemption_index        = 2,
        .global_prefill_tokens   = 4000,
        .global_decode_tokens    = 190,
        .global_replayed_tokens  = 2048,
        .request_prefill_tokens  = 192,
        .request_decode_tokens   = 17,
        .request_replayed_tokens = 512,
    };
    const Json scheduling_json = Json::parse(
        format_request_scheduling_json("serve-test", 3002, 8, "http-eight", scheduling));
    failures += check(
        scheduling_json.at("event") == "request_scheduling" &&
            scheduling_json.at("request").at("request_id") == 8 &&
            scheduling_json.at("request").at("http_request_id") == "http-eight" &&
            scheduling_json.at("engine_request_id") == 19 &&
            scheduling_json.at("transition") == "replay_complete" &&
            scheduling_json.at("route") == "replay" &&
            scheduling_json.at("steady_ns").get<std::uint64_t>() == scheduling.steady_ns &&
            scheduling_json.at("elapsed_ns") == scheduling.elapsed_ns &&
            scheduling_json.at("preemption_index") == 2 &&
            scheduling_json.at("progress").at("global_decode_tokens") == 190 &&
            scheduling_json.at("progress").at("request_decode_tokens") == 17 &&
            scheduling_json.at("progress").at("global_prefill_tokens") == 4000 &&
            scheduling_json.at("progress").at("request_prefill_tokens") == 192 &&
            scheduling_json.at("progress").at("global_replayed_tokens") == 2048 &&
            scheduling_json.at("progress").at("request_replayed_tokens") == 512,
        "scheduling observation lost request identity, precise clock or paired work counters");
    auto pausing       = scheduling;
    pausing.transition = ninfer::GenerationSchedulingTransition::PauseStarted;
    pausing.route      = ninfer::GenerationRecoveryRoute::None;
    const Json pausing_json =
        Json::parse(format_request_scheduling_json("serve-test", 3003, 8, "http-eight", pausing));
    failures += check(pausing_json.at("transition") == "pause_started" &&
                          pausing_json.at("route").is_null(),
                      "pause preparation claimed a completed Snapshot or Replay route");
    failures +=
        check(first_json.at("elapsed_seconds") == first.elapsed_seconds &&
                  first_json.at("initial_binding_seconds") == first.initial_binding_seconds &&
                  first_json.at("computed_prefill_tokens") == 192 &&
                  first_json.at("engine").at("units").at("prefill") == 3 &&
                  first_json.at("engine").at("host_exposed_seconds").at("session_call") == 0.0015 &&
                  first_json.at("engine").at("scheduler_yield_exposed_seconds") == 0.12 &&
                  first_json.at("scheduling").at("replayed_tokens") == 64 &&
                  first_json.at("scheduling").at("paused_ns") == 123456789 &&
                  observed_done.at("engine_timing") == done.at("engine_timing") &&
                  observed_done.at("generation").at("scheduling") ==
                      done.at("generation").at("scheduling"),
              "first-output snapshot lost its boundary or replaced terminal request statistics");
    failures += check(first_json.at("prefill").at("total").at("total_ns") == 9000000 &&
                          first_json.at("prefill").at("result").at("count") == 2 &&
                          first_json.at("replay").at("total").at("total_ns") == 21000000,
                      "request execution exposure merged replay and initial prefill");
    failures += check(
        first_json.at("context_transfers").at("state").at("d2h") ==
                Json{{"images", 0},
                     {"page_groups", 0},
                     {"copy_operations", 0},
                     {"bytes", 9007199254740993ULL},
                     {"seconds", 0.010}} &&
            first_json.at("context_transfers").at("kv_pools").at(0).at("transfers").at("h2d") ==
                Json{{"images", 0},
                     {"page_groups", 0},
                     {"copy_operations", 0},
                     {"bytes", 2113536},
                     {"seconds", 0.011}} &&
            first_json.at("context_transfers").at("kv_pools").at(1).at("transfers").at("d2d") ==
                Json{{"images", 0},
                     {"page_groups", 0},
                     {"copy_operations", 0},
                     {"bytes", 16384},
                     {"seconds", 0.012}} &&
            first_json.at("context_transfers").at("kv_pools").at(0).at("transfers").at("d2h") ==
                Json{{"images", 0},
                     {"page_groups", 0},
                     {"copy_operations", 0},
                     {"bytes", 0},
                     {"seconds", 0.0}},
        "first-output transfer resource, direction or integer precision changed");

    ThroughputReport window;
    window.interval_seconds                                                                 = 5;
    window.previous.speculative_draft_tokens                                                = 100;
    window.current.speculative_draft_tokens                                                 = 300;
    window.previous.speculative_accepted_tokens                                             = 50;
    window.current.speculative_accepted_tokens                                              = 110;
    window.current_execution[static_cast<std::size_t>(ninfer::ExecutionKind::Decode)].units = 2;
    window.current_execution[static_cast<std::size_t>(ninfer::ExecutionKind::Decode)].rows  = 4;
    window.current_execution[static_cast<std::size_t>(ninfer::ExecutionKind::Decode)]
        .exposure.total.total_ns = 20000000;
    const auto recent            = Json::parse(format_throughput_json("serve-test", 3002, window));
    failures += check(
        recent["speculative"]["acceptance"] == 0.3 &&
            recent["execution"]["decode"]["mean_seconds"] == 0.01 &&
            recent["execution"]["prefill"]["mean_seconds"].is_null(),
        "recent statistics must divide token deltas and batch counts, retaining missing samples");

    const OperationalRecord pretty_done = render_request_done(context, outcome);
    failures += check(
        pretty_done.message ==
            "req#7 done | openai-chat | output limit | prompt 401 | output 1,024 | cache 101 "
            "(25.2%, checkpoint) | TTFT 358 ms | total 5.7s | prefill 1.28k tok/s | "
            "decode 191.4 tok/s | mtp accepted 720/900 (80.0%) | thinking 256/256, control 19",
        "pretty request-done record mismatch");

    GenerationOutcome normalized_tool_outcome = outcome;
    normalized_tool_outcome.tool_calls.push_back(
        ninfer::GeneratedToolCall{.name = "Edit", .arguments_json = R"({"file_path":"x"})"});
    normalized_tool_outcome.tool_call_parse = {
        .marker_seen               = true,
        .structured_call_count     = 1,
        .schema_mismatch_arguments = 2,
        .fallback_reason           = ninfer::ToolCallParseFallbackReason::None,
    };
    const Json normalized_tool_done =
        Json::parse(format_request_done_json("serve-test", 3002, context, normalized_tool_outcome));
    failures += check(
        normalized_tool_done.at("result").at("tool_call_count") == 1 &&
            normalized_tool_done.at("result").at("tool_call_parse").at("structured_call_count") ==
                1 &&
            normalized_tool_done.at("result")
                    .at("tool_call_parse")
                    .at("schema_mismatch_arguments") == 2 &&
            normalized_tool_done.at("result").at("tool_call_parse").at("fallback_reason") ==
                "none" &&
            !render_tool_call_fallback(context, normalized_tool_outcome),
        "successful tool-call normalization diagnostics are incomplete or noisy");

    GenerationOutcome fallback_outcome = outcome;
    fallback_outcome.tool_call_parse   = {
          .marker_seen               = true,
          .structured_call_count     = 0,
          .schema_mismatch_arguments = 0,
          .fallback_reason           = ninfer::ToolCallParseFallbackReason::UndeclaredTool,
    };
    const Json fallback_done =
        Json::parse(format_request_done_json("serve-test", 3003, context, fallback_outcome));
    failures += check(fallback_done.at("result").at("tool_call_parse").at("marker_seen") &&
                          fallback_done.at("result").at("tool_call_parse").at("fallback_reason") ==
                              "undeclared_tool",
                      "tool-call text fallback diagnostics missing from JSONL");
    const std::optional<OperationalRecord> fallback_warning =
        render_tool_call_fallback(context, fallback_outcome);
    failures += check(
        fallback_warning && fallback_warning->severity == OperationalSeverity::Warning &&
            fallback_warning->message == "req#7 tool markup returned as text | undeclared tool",
        "tool-call text fallback warning is absent or exposes raw content");

    const Json error =
        Json::parse(format_request_error_json("serve-test", 4000, context, "generation failed"));
    failures += check(error.at("event") == "request_error", "request error event mismatch");
    failures += check(error.at("error").at("message") == "generation failed",
                      "request error message missing");

    const OperationalRecord internal_failure = render_request_failure(
        context,
        make_internal_request_failure(RequestFailurePhase::Generation, "sentinel-internal-detail"));
    failures +=
        check(internal_failure.severity == OperationalSeverity::Error &&
                  internal_failure.message.find("sentinel-internal-detail") == std::string::npos,
              "operational internal failure severity or data policy mismatch");
    const OperationalRecord disconnected = render_request_failure(
        context, make_client_disconnected_failure(RequestFailurePhase::Transport));
    failures += check(disconnected.severity == OperationalSeverity::Info &&
                          disconnected.message.find("req#7 cancelled during transport") !=
                              std::string::npos,
                      "client disconnect is not an informational cancellation");

    ThroughputReport throughput;
    throughput.interval_seconds                           = 2.0;
    throughput.computed_prefill_tokens                    = 100;
    throughput.committed_decode_tokens                    = 40;
    throughput.decode_rounds                              = 10;
    throughput.decode_row_rounds                          = 18;
    throughput.previous.root_selections                   = 2;
    throughput.previous.checkpoint_selections             = 3;
    throughput.previous.preemptions                       = 8;
    throughput.previous.snapshot_restores                 = 4;
    throughput.previous.replay_restores                   = 2;
    throughput.previous.replayed_tokens                   = 128;
    throughput.previous.context_transfers.state[1].bytes  = 100;
    throughput.current.running_requests                   = 2;
    throughput.current.prefilling_requests                = 1;
    throughput.current.decode_ready_requests              = 1;
    throughput.current.waiting_requests                   = 3;
    throughput.current.paused_requests                    = 2;
    throughput.current.replaying_requests                 = 1;
    throughput.current.preemptions                        = 11;
    throughput.current.snapshot_restores                  = 5;
    throughput.current.replay_restores                    = 4;
    throughput.current.replayed_tokens                    = 896;
    throughput.current.materializing_requests             = 1;
    throughput.current.capture_pending_requests           = 1;
    throughput.current.terminal_pending_requests          = 1;
    throughput.current.root_selections                    = 3;
    throughput.current.checkpoint_selections              = 5;
    throughput.current.context_transfers.state[1].images  = 1;
    throughput.current.context_transfers.state[1].bytes   = 132;
    throughput.current.context_transfers.state[1].seconds = 0.25;
    throughput.current.state_objects                      = 3;
    throughput.current.context_pools                      = {{.rank                = 0,
                                                              .domain              = ninfer::ContextMemoryDomain::Host,
                                                              .capacity_bytes      = 64ULL << 20,
                                                              .live_bytes          = 12ULL << 20,
                                                              .reserved_bytes      = 4ULL << 20,
                                                              .peak_occupied_bytes = 20ULL << 20}};
    throughput.current.last_selected_frontier_tokens      = 64;
    throughput.current.pressure_spill_pages               = 4;
    throughput.current.host_work                          = {
                                 .engine_boundary_ns            = 1000000,
                                 .session_call_ns               = 2000000,
                                 .executor_poll_ns              = 3000000,
                                 .engine_commit_output_ns       = 4000000,
                                 .engine_maintenance_ns         = 5000000,
                                 .scheduler_yield_ns            = 300000000,
                                 .decode_host_ns                = 10000000,
                                 .decode_scheduler_yield_ns     = 200000000,
                                 .prefill_host_ns               = 3000000,
                                 .prefill_scheduler_yield_ns    = 50000000,
                                 .control_host_ns               = 2000000,
                                 .control_scheduler_yield_ns    = 50000000,
                                 .prefill_units                 = 4,
                                 .control_units                 = 1,
                                 .stats_publication_ns          = 250000,
                                 .stats_publication_invocations = 5,
    };
    const std::string pretty_throughput = render_throughput(throughput).message;
    failures +=
        check(pretty_throughput ==
                  "throughput | 2.0s | prefill 50.0 tok/s (100 tok) | decode 20.0 tok/s (40 tok) | "
                  "running 2 (prefill 1, decode-ready 1) | waiting 3 | paused 2 | replaying 1 | "
                  "materializing 1 | "
                  "capture-pending 1 | terminal-pending 1 | batch 1.80 | host 0.8% (15.0 ms)",
              "pretty throughput record mismatch");
    ThroughputReport single_decode;
    single_decode.interval_seconds                     = 5.000168;
    single_decode.committed_decode_tokens              = 1025;
    single_decode.decode_rounds                        = 1025;
    single_decode.decode_row_rounds                    = 1025;
    single_decode.current.running_requests             = 1;
    single_decode.current.decode_ready_requests        = 1;
    single_decode.current.host_work.engine_boundary_ns = 69'241'000;
    const std::string single_decode_pretty             = render_throughput(single_decode).message;
    failures += check(
        single_decode_pretty ==
                "throughput | 5.0s | decode 205.0 tok/s (1,025 tok) | running 1 (decode-ready 1) | "
                "batch 1.00 | host 1.4% (69.2 ms)" &&
            single_decode_pretty.find("prefill") == std::string::npos &&
            single_decode_pretty.find("waiting") == std::string::npos,
        "single-request pretty throughput is noisy or incomplete");
    const Json throughput_json =
        Json::parse(format_throughput_json("serve-test", 5000, throughput));
    failures += check(throughput_json.at("event") == "throughput", "throughput event mismatch");
    failures += check(!throughput_json.at("final_interval").get<bool>(),
                      "periodic interval was marked as shutdown tail");
    ThroughputReport tail = throughput;
    tail.final_interval   = true;
    tail.interval_seconds = 0.125;
    const Json tail_json  = Json::parse(format_throughput_json("serve-test", 5125, tail));
    failures += check(tail_json.at("final_interval").get<bool>() &&
                          tail_json.at("interval_seconds") == 0.125 &&
                          tail_json.at("context_cache")
                                  .at("occupancy")
                                  .at("pools")
                                  .at(0)
                                  .at("peak_occupied_bytes") == (20ULL << 20),
                      "shutdown interval or allocator high-water mark was lost");
    failures += check(throughput_json.at("tokens").at("computed_prefill") == 100 &&
                          throughput_json.at("tokens").at("committed_decode") == 40,
                      "throughput token deltas mismatch");
    failures += check(throughput_json.at("decode_batch").at("average_size") == 1.8,
                      "throughput batch average mismatch");
    failures += check(throughput_json.at("scheduler").at("materializing") == 1 &&
                          throughput_json.at("scheduler").at("paused") == 2 &&
                          throughput_json.at("scheduler").at("replaying") == 1 &&
                          throughput_json.at("scheduler").at("capture_pending") == 1 &&
                          throughput_json.at("scheduler").at("terminal_pending") == 1,
                      "context scheduler gauges missing");
    failures += check(throughput_json.at("scheduling") == Json{{"preemptions", 3},
                                                               {"snapshot_restores", 1},
                                                               {"replay_restores", 2},
                                                               {"replayed_tokens", 768}},
                      "scheduling counters were not reported as interval deltas");
    failures += check(
        throughput_json.at("context_cache").at("occupancy").at("pools").at(0).at("live_bytes") ==
                (12ULL << 20) &&
            throughput_json.at("context_cache")
                    .at("occupancy")
                    .at("pools")
                    .at(0)
                    .at("reserved_bytes") == (4ULL << 20),
        "unified Host occupied and reserved gauges missing");
    failures += check(
        std::abs(throughput_json.at("host_work").at("elapsed_seconds").at("total").get<double>() -
                 0.015) < 1.0e-15 &&
            std::abs(throughput_json.at("host_work").at("scheduler_yield_seconds").get<double>() -
                     0.3) < 1.0e-15 &&
            std::abs(throughput_json.at("host_work")
                         .at("decode_host_microseconds_per_round")
                         .get<double>() -
                     1000.0) < 1.0e-12 &&
            std::abs(throughput_json.at("host_work")
                         .at("decode_scheduler_yield_microseconds_per_round")
                         .get<double>() -
                     20000.0) < 1.0e-12 &&
            std::abs(throughput_json.at("host_work")
                         .at("detail_microseconds_per_invocation")
                         .at("stats_publication")
                         .get<double>() -
                     50.0) < 1.0e-12,
        "throughput Host work deltas or normalization are incorrect");

    ThroughputReport zero_rounds;
    const Json zero_rounds_json =
        Json::parse(format_throughput_json("serve-test", 5001, zero_rounds));
    failures += check(
        zero_rounds_json.at("decode_batch").at("average_size").is_null() &&
            zero_rounds_json.at("host_work").at("decode_host_microseconds_per_round").is_null() &&
            zero_rounds_json.at("host_work")
                .at("detail_microseconds_per_invocation")
                .at("stats_publication")
                .is_null(),
        "zero Host-work denominators must serialize as null");
    failures += check(
        throughput_json.at("context_cache").at("selections").at("root") == 1 &&
            throughput_json.at("context_cache").at("state_transfers").at("h2d").at("bytes") == 32 &&
            throughput_json.at("context_cache").at("occupancy").at("state_objects") == 3 &&
            throughput_json.at("context_cache").at("pressure").at("spill_pages") == 4 &&
            throughput_json.at("context_cache").at("selections").at("checkpoint") == 2,
        "context-cache throughput statistics missing or not interval-scoped");

    const std::filesystem::path log_path =
        std::filesystem::temp_directory_path() /
        ("ninfer-request-log-test-" + std::to_string(static_cast<long long>(::getpid())) +
         ".jsonl");
    std::filesystem::remove(log_path);
    {
        JsonlRequestLog writer(log_path.string());
        writer.write_request_start(context);
    }
    {
        JsonlRequestLog writer(log_path.string());
        writer.write_request_rejected(rejected_context);
        writer.write_request_error(context, "generation failed");
    }
    std::ifstream input(log_path);
    std::string first_line;
    std::string second_line;
    std::string third_line;
    std::string extra_line;
    std::getline(input, first_line);
    std::getline(input, second_line);
    std::getline(input, third_line);
    std::getline(input, extra_line);
    failures += check(!first_line.empty() && !second_line.empty() && !third_line.empty() &&
                          extra_line.empty(),
                      "JSONL writer did not append exactly one flushed line per event");
    if (!first_line.empty() && !second_line.empty() && !third_line.empty()) {
        failures += check(Json::parse(first_line).at("event") == "request_start",
                          "first appended event mismatch");
        failures += check(Json::parse(second_line).at("event") == "request_rejected",
                          "second appended event mismatch");
        failures += check(Json::parse(third_line).at("event") == "request_error",
                          "third appended event mismatch");
    }
    input.close();
    std::filesystem::remove(log_path);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
