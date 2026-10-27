#include "models/qwen4_exp/program/graph_preparation.h"
#include "models/qwen4_exp/program/internal.h"
#include "core/startup.h"

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
    auto plan = plan_graph_probe(
        parameters.profile,
        std::get<std::shared_ptr<const ple::disk::Table>>(parameters.model.ple_table())->geometry(),
        options, device, moe);
    auto probe    = std::make_unique<detail::RankProgramImpl>(parameters, std::move(plan), device);
    result.memory = begin_graph_calibration(
        device, [&](GraphPreparationMode mode) { return probe->initialize_graphs(mode); },
        [&] { probe->release_graphs(); });
    probe.reset();
    device.synchronize();
    plan = plan_graph_probe(
        parameters.profile,
        std::get<std::shared_ptr<const ple::disk::Table>>(parameters.model.ple_table())->geometry(),
        options, device, moe, GraphProbeBacking::StartupWorkingSet);
    probe = std::make_unique<detail::RankProgramImpl>(parameters, std::move(plan), device);
    complete_graph_calibration(
        result.memory, device,
        [&](GraphPreparationMode mode) { return probe->initialize_graphs(mode); },
        [&] { probe->release_graphs(); });
    probe.reset();
    device.synchronize();
    phase.complete();
    return result;
}

} // namespace ninfer::models::qwen4_exp
