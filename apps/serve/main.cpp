#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <iostream>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const ninfer::serve::ServeOptions options = ninfer::serve::parse_serve_options(argc, argv);
        if (options.help_requested) {
            std::cout << ninfer::serve::serve_usage_text(argv[0]);
            return 0;
        }

        using Clock = std::chrono::steady_clock;
        std::cerr << "ninfer-serve: loading model...\n";
        const auto load_start = Clock::now();
        ninfer::serve::GenerationService service(options);
        std::cerr << "ninfer-serve: model loaded in "
                  << std::chrono::duration<double>(Clock::now() - load_start).count() << " s\n";

        std::cerr << "ninfer-serve: warming up...\n";
        service.warmup();

        ninfer::serve::HttpServer server(service, options);
        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        std::cerr << "ninfer-serve: listening on http://" << options.host << ':' << options.port
                  << " (model id: " << options.model_id
                  << ", auth: " << (options.api_key.empty() ? "disabled" : "bearer") << ")\n";

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            std::cerr << "error: failed to bind " << options.host << ':' << options.port << '\n';
            return 1;
        }
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "error: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    }
}
