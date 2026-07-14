#include "ninfer/serve/generation_service.h"
#include "ninfer/serve/http_server.h"
#include "ninfer/serve/serve_options.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

std::string format_bytes(std::uint64_t bytes) {
    constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
    constexpr double kMiB = 1024.0 * 1024.0;
    std::ostringstream out;
    out << std::fixed << std::setprecision(2);
    if (bytes >= static_cast<std::uint64_t>(kGiB)) {
        out << static_cast<double>(bytes) / kGiB << " GiB";
    } else if (bytes >= static_cast<std::uint64_t>(kMiB)) {
        out << static_cast<double>(bytes) / kMiB << " MiB";
    } else {
        out << bytes << " B";
    }
    return out.str();
}

std::string format_kv_dtype(ninfer::DType dtype) {
    switch (dtype) {
    case ninfer::DType::BF16:
        return "bf16";
    case ninfer::DType::I8:
        return "int8";
    default:
        return "unknown";
    }
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
        const ninfer::EngineMemoryStats memory = service.memory_stats();
        std::cerr << "ninfer-serve: kv cache dtype=" << format_kv_dtype(memory.kv_dtype) << " payload="
                  << format_bytes(static_cast<std::uint64_t>(memory.kv_cache_payload_bytes))
                  << " cache_used="
                  << format_bytes(static_cast<std::uint64_t>(memory.cache.used_bytes)) << " / "
                  << format_bytes(static_cast<std::uint64_t>(memory.cache.capacity_bytes)) << '\n';

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
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    }
}
