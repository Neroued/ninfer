#pragma once

#include "models/common/moe/schedule.h"
#include "models/common/graph_preparation.h"
#include "ninfer/types.h"

namespace ninfer::models::qwen3_5 {
namespace execution {
class Parameters;
}

struct GraphPreparationPlan {
    bool segmented = false;
    std::vector<moe::GraphPlan> layers_by_batch;
    friend bool operator==(const GraphPreparationPlan&, const GraphPreparationPlan&) = default;
};

struct GraphCalibration {
    GraphPreparationPlan plan;
    GraphMemorySummary memory;
};

[[nodiscard]] GraphPreparationPlan describe_graph_preparation(const EngineOptions&,
                                                              const moe::Plan*);
// The loaded model outlives calibration. No probe resources survive this call.
[[nodiscard]] GraphCalibration calibrate_graphs(const execution::Parameters&, DeviceContext&,
                                                const EngineOptions&,
                                                std::shared_ptr<const moe::Plan>);

} // namespace ninfer::models::qwen3_5
