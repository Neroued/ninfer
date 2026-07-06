#pragma once

#include "qus/serve/generation_service.h"
#include "qus/serve/serve_options.h"

#include <httplib.h>

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

    GenerationService& service_;
    ServeOptions options_;
    httplib::Server server_;
};

} // namespace qus::serve
