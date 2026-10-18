#include "runtime/engine/startup.h"
#include "core/platform.h"
#include "runtime/engine/startup.inl"
#include "models/qwen3_5/startup.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen4_exp/startup.h"

#include <atomic>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace ninfer::runtime {
namespace {
using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    if (options.enable_vision && options.media_live_bytes == 0) {
        throw std::invalid_argument(
            "Engine media_live_bytes must be nonzero when Vision is enabled");
    }
    if (options.media_preprocess_threads > 64) {
        throw std::invalid_argument("Engine media_preprocess_threads must be in [0,64]");
    }
}

std::size_t device_budget(std::size_t available) {
    const char* value = std::getenv("NINFER_VRAM_BUDGET_MIB");
    if (!value) return available;
    const std::string_view text(value);
    std::uint64_t mib = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), mib);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !mib ||
        mib > std::numeric_limits<std::size_t>::max() / (1024ULL * 1024)) {
        throw std::invalid_argument("NINFER_VRAM_BUDGET_MIB requires a positive MiB integer");
    }
    return std::min(available, static_cast<std::size_t>(mib * 1024 * 1024));
}

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.kv_capacity          = KvCapacityPolicy::explicit_capacity(options.max_context);
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache        = ContextCacheOptions{
                   .enabled = false, .device_state_slots = 0, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    ContextCacheOptions& cache = options.context_cache;
    if constexpr (NINFER_SHARED_PHYSICAL_MEMORY != 0) cache.host_capacity_bytes = 0;
    const std::uint32_t concurrency = options.max_concurrency;
    cache.device_state_slots        = cache.device_state_slots.value_or(concurrency);
    const std::uint64_t total_device_state_slots =
        static_cast<std::uint64_t>(concurrency) + *cache.device_state_slots;
    if (total_device_state_slots > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("context cache Device state capacity exceeds uint32");
    }
    return options;
}

SessionEpoch next_session_epoch() noexcept {
    static std::atomic<std::uint64_t> next{0};
    return {next.fetch_add(1, std::memory_order_relaxed) + 1};
}

ConstructedModel construct_model(EngineOptions& options) {
    validate_options(options);
    std::unique_ptr<DeviceContext> device_owner;
    {
        StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaInitialize);
        device_owner = std::make_unique<DeviceContext>(options.device);
        phase.complete();
    }
    auto& device       = *device_owner;
    const auto start   = Clock::now();
    const auto initial = device_memory_snapshot(device.device());
    const auto limit   = device_budget(initial.free_bytes);
    const auto headroom =
        std::max(kDefaultKvCapacityHeadroomBytes, options.kv_capacity.automatic_headroom_bytes);
    // Retained and in-flight prepared media share the live-byte account.
    const auto host_reservation = (NINFER_SHARED_PHYSICAL_MEMORY != 0) && options.enable_vision
                                      ? options.media_live_bytes
                                      : 0;
    StartupMemoryBudget budget(initial, limit, headroom, host_reservation);
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    switch (models::resolve_architecture(reader.directory())) {
    case models::Architecture::Qwen4Exp: {
        auto built = detail::build_model<models::qwen4_exp::Startup>(
            reader, options, std::move(device_owner), budget, start);
        return {std::move(built.instance), std::move(built.load), std::move(built.context_cost)};
    }
    case models::Architecture::Qwen3_5:
    case models::Architecture::Qwen3_5Moe: {
        auto built = detail::build_model<models::qwen3_5::Startup>(
            reader, options, std::move(device_owner), budget, start);
        return {std::move(built.instance), std::move(built.load), std::move(built.context_cost)};
    }
    }
    throw std::logic_error("architecture has no Engine construction route");
}

} // namespace ninfer::runtime
