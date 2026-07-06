#include "qus/serve/generation_service.h"
#include "qus/serve/http_server.h"
#include "qus/serve/serve_options.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>

namespace {

std::atomic<qus::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    qus::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const qus::serve::ServeOptions options = qus::serve::parse_serve_options(argc, argv);
        if (options.help_requested) {
            std::cout << qus::serve::serve_usage_text(argv[0]);
            return 0;
        }

        using Clock = std::chrono::steady_clock;
        std::cerr << "qus-serve: loading model...\n";
        const auto load_start = Clock::now();
        qus::serve::GenerationService service(options);
        std::cerr << "qus-serve: model loaded in "
                  << std::chrono::duration<double>(Clock::now() - load_start).count() << " s\n";

        std::cerr << "qus-serve: warming up...\n";
        service.warmup();

        qus::serve::HttpServer server(service, options);
        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::cerr << "qus-serve: listening on http://" << options.host << ':' << options.port
                  << " (model id: " << options.model_id << ", auth: "
                  << (options.api_key.empty() ? "disabled" : "bearer") << ")\n";

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            std::cerr << "error: failed to bind " << options.host << ':' << options.port << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << qus::serve::serve_usage_text(argv[0]);
        return 1;
    }
}
