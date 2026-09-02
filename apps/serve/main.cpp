#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

const char* kv_capacity_mode_name(ninfer::KvCapacityMode mode) noexcept {
    return mode == ninfer::KvCapacityMode::Automatic ? "auto" : "explicit";
}

void log_engine_capacity(const std::shared_ptr<spdlog::logger>& logger,
                         const ninfer::serve::GenerationService& service,
                         const ninfer::serve::ServeOptions& options) {
    const ninfer::MemorySummary memory            = service.memory_summary();
    const ninfer::ContextCostSummary context_cost = service.load_summary().context_cost;
    const ninfer::EngineOptions& engine           = service.engine_options();
    const ninfer::ContextCacheOptions& cache      = engine.context_cache;
    logger->info(
        "engine capacity kv_capacity_mode={} kv_capacity_tokens={} kv_page_groups={} "
        "kv_max_page_groups={} runtime_reservation_bytes={} available_after_weights_bytes={} "
        "available_after_startup_bytes={} kv_headroom_bytes={} planned_slack_bytes={} "
        "cuda_graph_allowance_bytes={}",
        kv_capacity_mode_name(memory.kv_capacity_mode), memory.kv_capacity,
        memory.kv_capacity_page_groups, memory.kv_capacity_max_page_groups,
        memory.runtime_reservation_bytes, memory.available_after_weights_bytes,
        memory.available_after_startup_bytes, memory.kv_capacity_headroom_bytes,
        memory.planned_slack_bytes, memory.cuda_graph_allowance_bytes);
    logger->info("engine context_cache enabled={} active_lanes={} device_state_slots={} "
                 "host_state_slots={} host_kv_bytes={} private_continuations={} shared_prefixes={} "
                 "long_anchors_per_continuation={}",
                 cache.enabled, engine.max_concurrency, *cache.device_state_slots,
                 cache.host_state_slots, cache.host_kv_capacity_bytes,
                 *cache.max_private_continuations, *cache.max_shared_prefixes,
                 *cache.max_long_anchors_per_continuation);
    logger->info(
        "engine context_cost transfer_source={} prefill_source={} hardware_class={} model_id={} "
        "weights_id={}",
        ninfer::context_cost_preset_source_name(context_cost.transfer_source),
        ninfer::context_cost_preset_source_name(context_cost.prefill_source),
        ninfer::product::quote_log_value(context_cost.hardware_class),
        ninfer::product::quote_log_value(context_cost.model_id),
        ninfer::product::quote_log_value(context_cost.weights_id));
    if (options.enable_vision) {
        const ninfer::MediaCacheSummary media = service.media_cache_summary();
        logger->info(
            "engine media preprocess_threads={} cache_capacity_bytes={} live_capacity_bytes={}",
            media.preprocess_threads, media.capacity_bytes, media.live_capacity_bytes);
    }
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    ninfer::product::LoggingRuntime logging({.logger_name = "ninfer-serve"});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            logger->error("server status=failed phase=bind host={} port={}",
                          ninfer::product::quote_log_value(options.host), options.port);
            return 1;
        }

        ninfer::serve::GenerationService service(options, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        log_engine_capacity(logger, service, options);

        using Clock                            = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        logger->info("startup phase=serve-warmup status=begin");
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double duration_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - warmup_started).count();
            logger->error("startup phase=serve-warmup status=failed duration_ms={:.3f} detail={}",
                          duration_ms, ninfer::product::quote_log_value(exception.what()));
            throw;
        }
        logger->info(
            "startup phase=serve-warmup status=complete duration_ms={:.3f}",
            std::chrono::duration<double, std::milli>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        serving = true;
        logger->info("server status=ready host={} port={} model_id={} auth_enabled={}",
                     ninfer::product::quote_log_value(options.host), options.port,
                     ninfer::product::quote_log_value(server.public_model_id()),
                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            logger->error("server status=failed phase=listen host={} port={}",
                          ninfer::product::quote_log_value(options.host), options.port);
            return 1;
        }
        logger->info("server status=stopped");
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        logger->critical("server status=failed phase={} detail={}", serving ? "serving" : "startup",
                         ninfer::product::quote_log_value(exception.what()));
        return 1;
    }
}
