#pragma once

#include "serve/generation_service.h"
#include "serve/gpu_telemetry.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <memory>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::serve {

void write_openai_error(httplib::Response& response, const ApiError& error);
void write_anthropic_error(httplib::Response& response, const ApiError& error,
                           const std::string& request_id);

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

// True when `given` equals the API key; the time depends on the key's length only, not on where
// the first differing byte is (used for both Authorization: Bearer and x-api-key).
[[nodiscard]] bool api_key_equal(std::string_view given, std::string_view api_key) noexcept;

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;

// Browser-CSRF / DNS-rebinding guard helpers. Both return true for the "safe" case so that
// non-browser clients (which send no Origin and a loopback Host) are never rejected.
//   host_header_is_local:   true unless the HTTP Host header names a non-loopback authority.
//                           An empty Host counts as local.
//   origin_header_is_local: true unless the browser Origin header names a non-loopback origin.
//                           An empty Origin counts as local; "null" counts as non-local.
[[nodiscard]] bool host_header_is_local(std::string_view host_header) noexcept;
[[nodiscard]] bool origin_header_is_local(std::string_view origin_header) noexcept;

class HttpServer {
public:
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

    // True once the engine watch stopped listen() because the Engine reported an Engine-wide
    // failure; the process should then exit non-zero so a supervisor reloads the model.
    [[nodiscard]] bool engine_failed() const noexcept { return engine_failed_.load(); }

private:
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, RequestLogContext context);

        void done(const GenerationOutcome& outcome);
        void failure(const RequestFailure& failure);
        void response_failure(const RequestFailure& failure);

        [[nodiscard]] std::uint64_t request_id() const noexcept { return context_.id; }

    private:
        enum class State : std::uint8_t {
            Pending,
            Done,
            Error,
        };

        [[nodiscard]] bool claim(State terminal) noexcept;

        HttpServer* owner_ = nullptr;
        RequestLogContext context_;
        std::atomic<State> state_{State::Pending};
    };

    [[nodiscard]] std::shared_ptr<RequestLifecycle> begin_request(RequestLogContext context);

    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    void run_stats_reporter();
    void stop_stats_reporter();
    void run_engine_watch();
    void stop_engine_watch();

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    OpenAIResponsesStore openai_responses_store_;
    OperationalLog operational_log_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    // Owned by the statistics thread once listen() starts; created by attach() only when the
    // JSONL log and the statistics interval are both enabled.
    std::unique_ptr<GpuTelemetry> gpu_telemetry_;
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
    std::mutex watch_mutex_;
    std::condition_variable watch_cv_;
    std::thread watch_thread_;
    bool watch_stopping_ = false;
    std::atomic<bool> engine_failed_{false};
};

} // namespace ninfer::serve
