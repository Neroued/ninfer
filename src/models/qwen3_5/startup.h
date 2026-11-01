#pragma once

#include "models/common/startup.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/runtime_types.h"
#include "models/qwen3_5/program/graph_preparation.h"

namespace ninfer::models::qwen3_5 {

struct ExecutionPreparation {
    std::shared_ptr<const moe::Plan> experts;
};

// Native loading and execution planning. Runtime owns the budget, calibration loop and retries.
class Startup {
public:
    using Types         = RuntimeTypes;
    using ExecutionPlan = models::ExecutionPlan<SequencePlan, ExecutionPreparation>;
    using Calibration   = GraphCalibration;

    Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
            const EngineOptions&, Device);

    [[nodiscard]] Architecture architecture() const noexcept { return architecture_; }

    [[nodiscard]] std::optional<ExecutionPlan> plan(DeviceContext&, const EngineOptions&,
                                                    WeightPlacement, std::size_t available,
                                                    std::size_t available_host_bytes) const;
    [[nodiscard]] std::unique_ptr<ModelShard> materialize(DeviceContext&, WeightPlacement,
                                                          const StartupObserver&);
    [[nodiscard]] qwen::Frontend frontend(const EngineOptions&) const;
    [[nodiscard]] GraphPreparationPlan graph_plan(const EngineOptions&,
                                                  const ExecutionPreparation&) const;
    [[nodiscard]] Calibration calibrate(const execution::Parameters&, DeviceContext&,
                                        const EngineOptions&, const ExecutionPreparation&) const;

    [[nodiscard]] std::span<const WeightCandidate> placements() const { return placements_; }

    [[nodiscard]] std::string prefill_signature(const ExecutionPreparation&,
                                                std::uint32_t chunk) const;

    [[nodiscard]] const std::shared_ptr<const ModelDefinition>& definition() const {
        return definition_;
    }

    std::shared_ptr<const execution::ModelProfile> profile;

private:
    std::vector<WeightCandidate> placements_;
    std::shared_ptr<const moe::Plan> offload_;
    const artifact::Reader& reader_;
    std::shared_ptr<const ModelDefinition> definition_;
    Architecture architecture_;
};

} // namespace ninfer::models::qwen3_5
