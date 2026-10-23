#include "runtime/engine/startup.h"
#include "runtime/engine/startup.inl"
#include "models/qwen3_5/startup.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen4_exp/startup.h"
#if NINFER_ENABLE_DISTRIBUTED
#    include "runtime/engine/distributed_startup.inl"
#    include "models/qwen4_exp/parallel/startup.h"
#endif

#include <atomic>
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

} // namespace

EngineOptions normalize_engine_options(EngineOptions options) {
    switch (options.purpose) {
    case EnginePurpose::Generation:
        break;
    case EnginePurpose::CausalScoring:
        options.max_concurrency      = 1;
        options.max_pending_requests = 1;
        options.speculative          = {};
        options.enable_vision        = false;
        options.use_cuda_graph       = false;
        options.context_cache = ContextCacheOptions{.enabled = false, .host_capacity_bytes = 0};
        break;
    default:
        throw std::invalid_argument("Engine purpose is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,8]");
    }

    if (!options.context_cache.checkpoint_interval)
        throw std::invalid_argument("checkpoint interval must be positive");
    return options;
}

SessionEpoch next_session_epoch() noexcept {
    static std::atomic<std::uint64_t> next{0};
    return {next.fetch_add(1, std::memory_order_relaxed) + 1};
}

ConstructedModel construct_model(EngineOptions& options) {
    validate_options(options);
    if (options.deployment) {
        validate_deployment(*options.deployment);
        if (options.deployment->ranks.size() == 1)
            options.device = options.deployment->ranks.front().device;
    }
    const auto start = Clock::now();
    StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
    artifact::Reader reader(options.artifact_path);
    inspect.complete();
    switch (models::resolve_architecture(reader.directory())) {
    case models::Architecture::Qwen4Exp: {
#if NINFER_ENABLE_DISTRIBUTED
        if (options.deployment && options.deployment->ranks.size() > 1) {
            auto built = detail::build_distributed_model<models::qwen4_exp::ParallelStartup>(
                reader, options, start);
            return {std::move(built.instance), std::move(built.load),
                    std::move(built.context_cost)};
        }
#endif
        auto built = detail::build_local_model<models::qwen4_exp::Startup>(reader, options, start);
        return {std::move(built.instance), std::move(built.load), std::move(built.context_cost)};
    }
    case models::Architecture::Qwen3_5:
    case models::Architecture::Qwen3_5Moe: {
        auto built = detail::build_local_model<models::qwen3_5::Startup>(reader, options, start);
        return {std::move(built.instance), std::move(built.load), std::move(built.context_cost)};
    }
    }
    throw std::logic_error("architecture has no Engine construction route");
}

} // namespace ninfer::runtime
