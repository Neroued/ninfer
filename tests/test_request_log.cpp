#include "serve/request_log.h"

#include <nlohmann/json.hpp>

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
    options.artifact_path             = "/models/qwen3_6_27b.ninfer";
    options.host                      = "127.0.0.1";
    options.port                      = 8123;
    options.api_key                   = "must-not-appear";
    options.model_id                  = "qwen3.6-27b";
    options.request_log_jsonl         = "requests.jsonl";
    options.max_context               = 262144;
    options.prefill_chunk             = 1024;
    options.kv_cache                  = ninfer::KvCacheStorage::Int8Group64;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = false;
    options.allow_prefix_reuse        = false;
    options.startup_argv = {"ninfer-serve", options.artifact_path, "--api-key", "<redacted>"};

    ninfer::LoadSummary load;
    load.target               = "qwen3_6_27b";
    load.load_seconds         = 1.234567890123;
    load.upload_seconds       = 0.345678901234;
    load.artifact_bytes_read  = 1000;
    load.host_to_device_bytes = 900;
    load.peak_staging_bytes   = 128;
    load.tensor_count         = 42;
    load.resource_count       = 6;

    ninfer::MemorySummary memory;
    memory.max_context              = 262144;
    memory.kv_cache                 = ninfer::KvCacheStorage::Int8Group64;
    memory.weights.capacity_bytes   = 100;
    memory.sequence.capacity_bytes  = 200;
    memory.workspace.capacity_bytes = 300;
    memory.kv_payload_bytes         = 400;

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
        "serve-test", 1000, options, load, memory, environment, std::uint64_t{123456}));
    failures += check(server.at("artifact_type") == kRequestLogArtifactType,
                      "server record artifact type mismatch");
    failures += check(server.at("schema_version") == kRequestLogSchemaVersion,
                      "server record schema mismatch");
    failures += check(server.at("event") == "server_start", "server event mismatch");
    failures += check(server.at("artifact").at("target") == "qwen3_6_27b", "server target missing");
    failures += check(server.at("artifact").at("size_bytes") == 123456, "artifact size missing");
    failures += check(server.at("engine").at("max_context") == 262144, "max context missing");
    failures += check(server.at("server").at("request_log_jsonl") == "requests.jsonl",
                      "request log path missing");
    failures += check(server.at("engine").at("kv_cache") == "int8-group64", "KV type missing");
    failures += check(server.at("engine").at("vision") == false, "Vision state missing");
    failures += check(server.at("engine").at("speculative_backend") == "mtp",
                      "speculative backend missing");
    failures +=
        check(server.at("engine").at("proposal_head") == "optimized", "proposal head missing");
    failures +=
        check(server.at("engine").at("prefix_reuse") == false, "prefix-reuse state missing");
    failures += check(server.at("environment").at("gpu_name") == "NVIDIA GeForce RTX 5090",
                      "GPU name missing");
    failures += check(server.dump().find("must-not-appear") == std::string::npos,
                      "server JSON leaked the API key");
    failures += check(server.at("argv").at(3) == "<redacted>",
                      "server argv did not retain the redaction marker");

    GenerationRequest request;
    request.model          = "qwen3.6-27b";
    request.stream         = false;
    request.max_tokens     = 4096;
    request.max_tokens_set = true;
    request.messages.resize(2);

    PreparedRequest prepared;
    prepared.enable_thinking                              = false;
    prepared.options.execution.sampling.temperature       = 0.6F;
    prepared.options.execution.sampling.top_p             = 0.95F;
    prepared.options.execution.sampling.top_k             = 20;
    prepared.options.execution.sampling.min_p             = 0.0F;
    prepared.options.execution.sampling.presence_penalty  = 1.0F;
    prepared.options.execution.sampling.frequency_penalty = 0.0F;
    prepared.options.execution.sampling.seed              = 7632647173703958409ULL;

    const RequestLogContext context =
        make_request_log_context(7, "openai_chat_completions", request, prepared);
    const Json started = Json::parse(format_request_start_json("serve-test", 2000, context));
    failures +=
        check(started.at("request").at("request_id") == 7, "request id missing from start record");
    failures += check(started.at("request").at("requested_output_tokens") == 4096,
                      "request output budget missing");
    failures += check(started.at("request").at("enable_thinking") == false,
                      "resolved thinking mode missing");
    failures += check(started.at("request").at("sampling").at("seed") == 7632647173703958409ULL,
                      "resolved seed missing");

    GenerationOutcome outcome;
    outcome.prompt_tokens                             = 401;
    outcome.completion_tokens                         = 1024;
    outcome.finish_reason                             = ninfer::FinishReason::OutputLimit;
    outcome.metrics.prepare_seconds                   = 0.1234567890123;
    outcome.metrics.vision_seconds                    = 0.0;
    outcome.metrics.prefill_seconds                   = 0.2345678901234;
    outcome.metrics.decode_seconds                    = 5.3456789012345;
    outcome.metrics.total_seconds                     = 5.7037035803702;
    outcome.metrics.prefix_cache_hit_tokens           = 0;
    outcome.metrics.speculative_backend               = ninfer::SpeculativeBackend::Mtp;
    outcome.metrics.speculative_draft_window          = 3;
    outcome.metrics.speculative_rounds                = 300;
    outcome.metrics.speculative_draft_tokens          = 900;
    outcome.metrics.speculative_accepted_tokens       = 720;
    outcome.metrics.speculative_fallback_steps        = 2;
    outcome.metrics.speculative_accepted_per_position = {290, 240, 190};

    const Json done = Json::parse(format_request_done_json("serve-test", 3000, context, outcome));
    failures +=
        check(done.at("result").at("finish_reason") == "output_limit", "finish reason missing");
    failures += check(done.at("result").at("prompt_tokens") == 401, "prompt tokens missing");
    failures += check(done.at("timings_seconds").at("decode").get<double>() ==
                          outcome.metrics.decode_seconds,
                      "decode time lost precision");
    failures += check(done.at("speculative").at("backend") == "mtp", "speculative backend missing");
    failures +=
        check(done.at("speculative").at("draft_window") == 3, "speculative draft window missing");
    failures += check(done.at("speculative").at("fallback_steps") == 2,
                      "speculative fallback count missing");
    failures +=
        check(done.at("speculative").at("accepted_per_position") == Json::array({290, 240, 190}),
              "speculative position counts missing");

    const Json error =
        Json::parse(format_request_error_json("serve-test", 4000, context, "generation failed"));
    failures += check(error.at("event") == "request_error", "request error event mismatch");
    failures += check(error.at("error").at("message") == "generation failed",
                      "request error message missing");

    failures += check(format_request_start(context).find("thinking=off") != std::string::npos,
                      "human request log omits resolved thinking mode");

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
        writer.write_request_error(context, "generation failed");
    }
    std::ifstream input(log_path);
    std::string first_line;
    std::string second_line;
    std::string extra_line;
    std::getline(input, first_line);
    std::getline(input, second_line);
    std::getline(input, extra_line);
    failures += check(!first_line.empty() && !second_line.empty() && extra_line.empty(),
                      "JSONL writer did not append exactly one flushed line per event");
    if (!first_line.empty() && !second_line.empty()) {
        failures += check(Json::parse(first_line).at("event") == "request_start",
                          "first appended event mismatch");
        failures += check(Json::parse(second_line).at("event") == "request_error",
                          "second appended event mismatch");
    }
    input.close();
    std::filesystem::remove(log_path);

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
