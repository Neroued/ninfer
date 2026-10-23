#include "models/qwen3_5/startup.h"
#include "models/qwen3_5/execution/profile.h"
#include "models/qwen3_5/measurement.h"
#include "models/qwen3_5/program/planning/moe.h"
#include "core/platform.h"

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
            parallel.dp != 1 || parallel.pp != 1 || parallel.adp != 1)
            throw std::invalid_argument(
                "Qwen3_5 model preparation supports one rank with TP=EP=DP=PP=ADP=1");
    }
    profile       = execution::describe_model(*definition_);
    auto resident = resolve_load(reader_, definition_, device);
    placements_.push_back({WeightPlacement::Resident,
                           resident.materialization().device_capacity_bytes,
                           resident.materialization().host_capacity_bytes});
    if (architecture_ == Architecture::Qwen3_5Moe) {
        offload_ = detail::describe_moe_plan(*definition_, *profile, options.moe);
        if (offload_) {
            const auto host = resolve_load(reader_, definition_, device, Device::cpu());
            placements_.push_back({WeightPlacement::HostExperts,
                                   host.materialization().device_capacity_bytes,
                                   host.materialization().host_capacity_bytes});
        }
    }
}

std::optional<Startup::ExecutionPlan>
Startup::plan(DeviceContext& device, const EngineOptions& options, WeightPlacement placement,
              std::size_t available, std::size_t available_host_bytes) const {
    using runtime::resolve_context_capacity;
    const bool host_experts = placement == WeightPlacement::HostExperts;
    if (host_experts && !offload_) return {};
    auto trial       = options;
    const auto chunk = std::min(options.max_context, options.prefill_chunk ? options.prefill_chunk
                                                     : host_experts        ? 4096U
                                                                           : 1024U);
    trial.prefill_chunk             = chunk;
    auto policy                     = options.context_capacity;
    policy.automatic_headroom_bytes = 0;
    constexpr bool shared_memory    = NINFER_SHARED_PHYSICAL_MEMORY != 0;
    auto prototype                  = host_experts ? offload_ : nullptr;
    auto planner                    = make_sequence_planner(profile, device, trial, prototype);
    const auto& capacity_plan       = planner.capacity_plan();
    const auto required_context     = policy.mode == ContextCapacityMode::Explicit
                                          ? runtime::ContextCapacity{policy.explicit_bytes}
                                          : capacity_plan.minimum_capacity();
    const auto required_bytes       = capacity_plan.reservation_bytes(required_context) +
                                (shared_memory ? capacity_plan.control_bytes(required_context) : 0);
    if (required_bytes > available) return std::nullopt;
    if (!host_experts) {
        auto resolution = resolve_context_capacity(policy, capacity_plan, available,
                                                   available_host_bytes, shared_memory);
        return ExecutionPlan{
            std::move(planner).finalize(resolution.capacity()), resolution, placement, chunk, {}};
    }
    const auto max_tokens = static_cast<std::int32_t>(
        std::max(std::min(chunk, options.max_context),
                 options.max_concurrency * (options.speculative.draft_tokens + 1)));
    auto moe = offload_->resolve(available - required_bytes, max_tokens, options.moe.cpu_workers);
    if (!moe) return std::nullopt;
    auto final                 = make_sequence_planner(profile, device, trial, moe);
    const auto& final_capacity = final.capacity_plan();
    if (final_capacity.reservation_bytes(required_context) +
            (shared_memory ? final_capacity.control_bytes(required_context) : 0) >
        available)
        return std::nullopt;
    auto resolution = resolve_context_capacity(policy, final_capacity, available,
                                               available_host_bytes, shared_memory);
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
