#include "models/qwen4_exp/program/graph_preparation.h"
#include "models/qwen4_exp/program/internal.h"
#include "core/startup.h"

#include <algorithm>

namespace ninfer::models::qwen4_exp {

GraphPreparationPlan describe_graph_preparation(const EngineOptions& options,
                                                const moe::Plan* plan) {
    GraphPreparationPlan out;
    if (!options.use_cuda_graph) return out;
    out.enabled         = true;
    out.offload         = plan != nullptr;
    out.max_context     = options.max_context;
    out.max_concurrency = options.max_concurrency;
    out.draft_tokens    = options.speculative.draft_tokens;
    out.prefill_chunk   = options.prefill_chunk;
    out.kv_cache        = options.kv_cache;
    out.vision          = options.enable_vision;
    if (!plan) return out;
    for (std::uint32_t batch = 1; batch <= options.max_concurrency; ++batch)
        for (const auto& layer : plan->description().layers) {
            out.ordinary.push_back(plan->graph_plan(layer.bank, batch));
            if (options.speculative.backend == SpeculativeBackend::Mtp)
                out.verify.push_back(
                    plan->graph_plan(layer.bank, batch * (options.speculative.draft_tokens + 1)));
        }
    return out;
}

GraphCalibration calibrate_graphs(const execution::Parameters& parameters, DeviceContext& device,
                                  const EngineOptions& options,
                                  std::shared_ptr<const moe::Plan> moe) {
    GraphCalibration result;
    result.plan = describe_graph_preparation(options, moe.get());
    if (!options.use_cuda_graph) return result;
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaGraphCalibration);
    auto plan  = plan_graph_probe(parameters.profile, parameters.model.ple_table().geometry(),
                                  options, device, std::move(moe));
    auto probe = std::make_unique<detail::RankProgramImpl>(parameters, std::move(plan), device);
    const auto before = device_memory_snapshot(device.device());
    probe->initialize_graphs();
    const auto ready = device_memory_snapshot(device.device());
    probe->release_graphs();
    const auto released                = device_memory_snapshot(device.device());
    result.preparation_increment_bytes = device_memory_growth(before, ready);
    result.retained_increment_bytes    = device_memory_growth(before, released);
    result.rebuild_bytes =
        static_cast<std::size_t>(std::max<std::int64_t>(0, device_memory_growth(released, ready)));
    probe.reset();
    device.synchronize();
    phase.complete();
    return result;
}

} // namespace ninfer::models::qwen4_exp
