#pragma once

// Human-readable request summaries and the optional full-precision JSONL event log used for
// measurement. The HTTP layer owns request ids; this module owns one stable JSON schema and
// serializes concurrent writes from non-streaming handlers and streaming workers.

#include "serve/generation_service.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>

namespace ninfer::serve {

inline constexpr int kRequestLogSchemaVersion        = 2;
inline constexpr const char* kRequestLogArtifactType = "ninfer_serve_request_log";

struct RequestLogContext {
    std::uint64_t id = 0;
    std::string protocol;
    std::string model;
    bool stream                             = false;
    std::size_t message_count               = 0;
    int requested_output_tokens             = 0;
    bool requested_output_tokens_client_set = false;
    std::size_t tool_count                  = 0;
    ToolChoice tool_choice;
    bool has_tool_history = false;
    bool enable_thinking  = true;
    ninfer::SamplingParameters sampling;
};

struct ServerLogEnvironment {
    int device = 0;
    std::string gpu_name;
    std::string gpu_uuid;
    std::uint64_t total_device_memory_bytes = 0;
    int compute_capability_major            = 0;
    int compute_capability_minor            = 0;
    std::string cuda_compile_version;
    std::string cuda_runtime_version;
    std::string cuda_driver_version;
};

RequestLogContext make_request_log_context(std::uint64_t id, std::string protocol,
                                           const GenerationRequest& request,
                                           const PreparedRequest& prepared);

// Compact console records retained for operator visibility.
std::string format_request_start(const RequestLogContext& context);
std::string format_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
std::string format_request_error(const RequestLogContext& context, const std::string& message);

// Pure JSON formatters are public to repository tests. Each return value is one complete JSON
// object without a trailing newline.
std::string format_server_start_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms, const ServeOptions& options,
                                     const ninfer::LoadSummary& load,
                                     const ninfer::MemorySummary& memory,
                                     const ServerLogEnvironment& environment,
                                     std::optional<std::uint64_t> artifact_size_bytes);
std::string format_request_start_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context);
std::string format_request_done_json(const std::string& server_instance_id,
                                     std::uint64_t timestamp_unix_ms,
                                     const RequestLogContext& context,
                                     const GenerationOutcome& outcome);
std::string format_request_error_json(const std::string& server_instance_id,
                                      std::uint64_t timestamp_unix_ms,
                                      const RequestLogContext& context, const std::string& message);

ServerLogEnvironment query_server_log_environment(int device);

// Opens in append mode so one campaign file can contain multiple independently started MTP/model
// blocks. Every line carries server_instance_id because request ids restart at one per process.
class JsonlRequestLog {
public:
    explicit JsonlRequestLog(const std::string& path,
                             const std::string& protected_artifact_path = {});

    JsonlRequestLog(const JsonlRequestLog&)            = delete;
    JsonlRequestLog& operator=(const JsonlRequestLog&) = delete;

    [[nodiscard]] bool enabled() const noexcept { return output_.is_open(); }

    [[nodiscard]] const std::string& server_instance_id() const noexcept {
        return server_instance_id_;
    }

    void write_server_start(const ServeOptions& options, const ninfer::LoadSummary& load,
                            const ninfer::MemorySummary& memory);
    void write_request_start(const RequestLogContext& context);
    void write_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void write_request_error(const RequestLogContext& context, const std::string& message);

private:
    void append(std::string record);

    std::string path_;
    std::string server_instance_id_;
    std::ofstream output_;
    std::mutex mutex_;
    bool failed_ = false;
};

} // namespace ninfer::serve
