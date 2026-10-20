#include "models/qwen3_5/startup.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen3_5/program/planning/moe.h"

#include <algorithm>

namespace ninfer::models::qwen3_5 {
Startup::Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
                 const EngineOptions& options, Device device)
    : reader_(rank_source), definition_(read_definition(definition_source, load_options(options))),
      architecture_(definition_->config().text.architecture) {
    if (options.deployment) {
        const auto& deployment = *options.deployment;
        const auto& parallel   = deployment.parallel;
        if (deployment.ranks.size() != 1 || parallel.tp != 1 || parallel.ep != 1 ||
            parallel.dp != 1 || parallel.pp != 1)
            throw std::invalid_argument(
                "Qwen3_5 model preparation supports one rank with TP=EP=DP=PP=1");
    }
    profile       = execution::describe_model(*definition_);
    auto resident = resolve_load(reader_, definition_, device);
    placements_.push_back(
        {WeightPlacement::Resident, resident.materialization().device_capacity_bytes});
    if (architecture_ == Architecture::Qwen3_5Moe) {
        offload_ = detail::describe_moe_plan(*definition_, *profile, options.moe);
        if (offload_) {
            const auto host = resolve_load(reader_, definition_, device, Device::cpu());
            placements_.push_back(
                {WeightPlacement::HostExperts, host.materialization().device_capacity_bytes});
        }
    }
}

std::optional<Startup::ExecutionPlan> Startup::plan(DeviceContext& device,
                                                    const EngineOptions& options,
                                                    WeightPlacement placement,
                                                    std::size_t available) const {
    using runtime::resolve_context_capacity;
    const bool host_experts = placement == WeightPlacement::HostExperts;
    if (host_experts && !offload_) return {};
    auto trial       = options;
    const auto chunk = std::min(options.max_context, options.prefill_chunk ? options.prefill_chunk
                                                     : host_experts        ? 4096U
                                                                           : 1024U);
    trial.prefill_chunk             = chunk;
    auto policy                     = options.kv_capacity;
    policy.automatic_headroom_bytes = 0;
    auto prototype                  = host_experts ? offload_ : nullptr;
    auto planner                    = make_sequence_planner(profile, device, trial, prototype);
    const auto curve                = planner.capacity_curve();
    const auto page_tokens          = curve.primary_page_tokens;
    const auto pages =
        policy.mode == KvCapacityMode::Explicit
            ? static_cast<std::uint32_t>((std::uint64_t(policy.explicit_tokens) + page_tokens - 1) /
                                         page_tokens)
            : curve.minimum_primary_page_groups;
    const auto slots = std::uint64_t(options.max_concurrency) +
                       options.context_cache.extra_state_slots.value_or(0);
    if (slots > INT32_MAX) throw std::invalid_argument("StateImage slots exceed int32");
    if (curve.reservation_bytes({pages, static_cast<std::uint32_t>(slots)}) > available)
        return std::nullopt;
    if (!host_experts) {
        auto resolution = resolve_context_capacity(policy, options.context_cache, curve, available);
        return ExecutionPlan{
            std::move(planner).finalize(resolution.capacity()), resolution, placement, chunk, {}};
    }
    auto limited       = KvCapacityPolicy::explicit_capacity(curve.resolved_tokens(pages));
    auto initial_cache = options.context_cache;
    if (!initial_cache.extra_state_slots)
        initial_cache.extra_state_slots = initial_cache.enabled ? options.max_concurrency : 0;
    if (curve.reservation_bytes(
            {pages, options.max_concurrency + *initial_cache.extra_state_slots}) > available)
        return std::nullopt;
    auto base             = resolve_context_capacity(limited, initial_cache, curve, available);
    const auto max_tokens = static_cast<std::int32_t>(
        std::max(std::min(chunk, options.max_context),
                 options.max_concurrency * (options.speculative.draft_tokens + 1)));
    auto moe = offload_->resolve(available - base.runtime_reservation_bytes, max_tokens,
                                 options.moe.cpu_workers);
    if (!moe) return std::nullopt;
    auto final = make_sequence_planner(profile, device, trial, moe);
    if (final.capacity_curve().reservation_bytes(base.capacity()) > available) return std::nullopt;
    auto resolution =
        resolve_context_capacity(limited, options.context_cache, final.capacity_curve(), available);
    resolution.mode = policy.mode;
    return ExecutionPlan{std::move(final).finalize(resolution.capacity()),
                         resolution,
                         placement,
                         chunk,
                         {std::move(moe)}};
}

std::unique_ptr<ModelShard> Startup::materialize(DeviceContext& device, WeightPlacement placement,
                                                 const StartupObserver& observer) {
    auto resolved =
        resolve_load(reader_, definition_, device.device(),
                     placement == WeightPlacement::HostExperts ? Device::cpu() : device.device());
    return materialize_model(std::move(resolved), device, &observer);
}

qwen::Frontend Startup::frontend(const EngineOptions& options) const {
    return qwen::make_frontend(definition_->resources(),
                               {.chat_template_path       = options.chat_template_path,
                                .grammar_cache_bytes      = options.grammar_cache_bytes,
                                .vision_enabled           = options.enable_vision,
                                .max_context              = options.max_context,
                                .media_cache_bytes        = options.media_cache_bytes,
                                .media_live_bytes         = options.media_live_bytes,
                                .media_preprocess_threads = options.media_preprocess_threads},
                               default_sampling(architecture_));
}

GraphPreparationPlan Startup::graph_plan(const EngineOptions& options,
                                         const ExecutionPreparation& preparation) const {
    return describe_graph_preparation(options, preparation.experts.get());
}

Startup::Calibration Startup::calibrate(const execution::Parameters& parameters,
                                        DeviceContext& device, const EngineOptions& options,
                                        const ExecutionPreparation& preparation) const {
    return calibrate_graphs(parameters, device, options, preparation.experts);
}

std::string Startup::prefill_signature(const ExecutionPreparation& preparation,
                                       std::uint32_t chunk) const {
    return qwen3_5::prefill_signature(*definition_, preparation.experts.get(), chunk);
}

} // namespace ninfer::models::qwen3_5
