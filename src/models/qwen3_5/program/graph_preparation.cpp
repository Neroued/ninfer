#include "models/qwen3_5/program/graph_preparation.h"

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "core/startup.h"

namespace ninfer::models::qwen3_5 {

GraphPreparationPlan describe_graph_preparation(const EngineOptions& options, const moe::Plan* moe,
                                                std::span<const std::uint32_t> target_widths) {
    GraphPreparationPlan out;
    if (!options.use_cuda_graph) return out;
    out.target_widths.assign(target_widths.begin(), target_widths.end());
    if (out.target_widths.empty() && options.speculative.backend != SpeculativeBackend::None)
        out.target_widths.push_back(options.speculative.draft_tokens + 1);
    if (!moe) return out;
    out.segmented = true;
    const BatchExecutionPlan batches(make_batch_execution_spec(options.max_concurrency, 1, true));
    for (const auto batch : batches.graph_rows()) {
        for (const auto& layer : moe->description().layers) {
            out.layers_by_batch.push_back(
                moe->graph_plan(layer.bank, moe::decode_geometry(batches, batch, 1)));
            for (const auto width : out.target_widths) {
                out.layers_by_batch.push_back(
                    moe->graph_plan(layer.bank, moe::decode_geometry(batches, batch, width)));
            }
        }
    }
    return out;
}

GraphCalibration calibrate_graphs(const execution::Parameters& parameters, DeviceContext& device,
                                  const EngineOptions& options,
                                  std::shared_ptr<const moe::Plan> moe,
                                  std::span<const std::uint32_t> target_widths) {
    GraphCalibration result;
    result.plan = describe_graph_preparation(options, moe.get(), target_widths);
    if (!options.use_cuda_graph) return result;
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaGraphCalibration);
    auto plan =
        detail::make_graph_probe_plan(parameters.profile, device, options, moe, target_widths);
    auto probe =
        std::make_unique<detail::RankProgramImpl>(parameters, *plan, device, StartupObserver{});
    result.memory = begin_graph_calibration(
        device, [&](GraphPreparationMode mode) { return probe->initialize_graphs(mode); },
        [&] { probe->release_graphs(); });
    probe.reset();
    device.synchronize();
    plan  = detail::make_graph_probe_plan(parameters.profile, device, options, moe, target_widths,
                                          GraphProbeBacking::StartupWorkingSet);
    probe = std::make_unique<detail::RankProgramImpl>(parameters, *plan, device, StartupObserver{});
    complete_graph_calibration(
        result.memory, device,
        [&](GraphPreparationMode mode) { return probe->initialize_graphs(mode); },
        [&] { probe->release_graphs(); });
    probe.reset();
    device.synchronize();
    phase.complete();
    return result;
}

} // namespace ninfer::models::qwen3_5
