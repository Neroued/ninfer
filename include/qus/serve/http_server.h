#pragma once

#include "qus/serve/generation_service.h"
#include "qus/serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace qus::serve {

class HttpServer {
public:
    HttpServer(GenerationService& service, ServeOptions options);

    // Blocking listen loop. Returns false if the socket could not be bound.
    bool listen();
    void stop();

private:
    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;

    // Writes one console line ("qus-serve: <line>") under log_mutex_ so lines from
    // the httplib thread pool and streaming worker threads never interleave.
    void log_line(const std::string& line);

    GenerationService& service_;
    ServeOptions options_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex log_mutex_;
};

} // namespace qus::serve
