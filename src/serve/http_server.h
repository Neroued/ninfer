#pragma once

#include "serve/generation_service.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace ninfer::serve {

class HttpServer {
public:
    explicit HttpServer(ServeOptions options);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

private:
    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    // Writes one console line ("ninfer-serve: <line>") under log_mutex_ so lines from
    // the httplib thread pool and streaming worker threads never interleave.
    void log_line(const std::string& line);
    void log_request_start(const RequestLogContext& context);
    void log_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void log_request_error(const RequestLogContext& context, const std::string& message);

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex log_mutex_;
};

} // namespace ninfer::serve
