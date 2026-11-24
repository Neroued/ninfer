#include "models/common/graph_preparation.h"

#include <algorithm>

namespace ninfer::models {

GraphPreparation::GraphPreparation(DeviceContext& device, GraphPreparationMode mode)
    : device_(device), mode_(mode) {
    if (mode == GraphPreparationMode::Complete)
        throw std::logic_error("Graph preparation observation requires a single pass");
    device_.synchronize();
    before_ = device_memory_snapshot(device_.device());
}

void GraphPreparation::checkpoint() {
    device_.synchronize();
    sample_.increment_bytes =
        device_memory_growth(before_, device_memory_snapshot(device_.device()));
    sample_.peak_bytes =
        std::max(sample_.peak_bytes,
                 static_cast<std::size_t>(std::max<std::int64_t>(0, sample_.increment_bytes)));
}

GraphMemorySummary
begin_graph_calibration(DeviceContext& device,
                        const std::function<GraphPreparationSample(GraphPreparationMode)>& prepare,
                        const std::function<void()>& release) {
    const auto warmup  = prepare(GraphPreparationMode::Warmup);
    const auto initial = prepare(GraphPreparationMode::Capture);
    release();
    device.synchronize();
    return {.warmup_increment_bytes      = warmup.increment_bytes,
            .calibration_increment_bytes = initial.increment_bytes,
            .calibration_peak_bytes      = initial.peak_bytes};
}

void complete_graph_calibration(
    GraphMemorySummary& memory, DeviceContext& device,
    const std::function<GraphPreparationSample(GraphPreparationMode)>& prepare,
    const std::function<void()>& release) {
    const auto rebuild = prepare(GraphPreparationMode::Capture);
    release();
    device.synchronize();
    memory.rebuild_estimate_bytes  = rebuild.peak_bytes;
    memory.rebuild_increment_bytes = rebuild.increment_bytes;
}

} // namespace ninfer::models
