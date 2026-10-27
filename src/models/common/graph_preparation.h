#pragma once

#include "core/device.h"
#include "ninfer/observability/resources.h"

#include <functional>

namespace ninfer::models {

enum class GraphPreparationMode { Warmup, Capture, Complete };
enum class GraphProbeBacking { Inputs, StartupWorkingSet };

struct GraphPreparationSample {
    std::int64_t increment_bytes = 0;
    std::size_t peak_bytes       = 0;
};

// Samples only at native preparation boundaries where all submitted work has completed.
// The peak is an observed physical-capacity delta, not an allocation ownership counter.
class GraphPreparation {
public:
    GraphPreparation(DeviceContext&, GraphPreparationMode);

    [[nodiscard]] bool capturing() const noexcept { return mode_ == GraphPreparationMode::Capture; }

    void checkpoint();

    [[nodiscard]] GraphPreparationSample sample() const noexcept { return sample_; }
private:
    DeviceContext& device_;
    GraphPreparationMode mode_;
    DeviceMemorySnapshot before_;
    GraphPreparationSample sample_;
};

// The caller recreates backing between passes. Distributed owners synchronize successful
// allocation on all ranks before entering either pass; failures retain owners until abort.
[[nodiscard]] GraphMemorySummary
begin_graph_calibration(DeviceContext&,
                        const std::function<GraphPreparationSample(GraphPreparationMode)>& prepare,
                        const std::function<void()>& release);
void complete_graph_calibration(
    GraphMemorySummary&, DeviceContext&,
    const std::function<GraphPreparationSample(GraphPreparationMode)>& prepare,
    const std::function<void()>& release);

} // namespace ninfer::models
