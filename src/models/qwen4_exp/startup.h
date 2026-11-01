#pragma once

#include "models/common/startup.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/runtime_types.h"
#include "models/qwen4_exp/program/graph_preparation.h"
#include "models/registry.h"

namespace ninfer::models::qwen4_exp {

struct ExecutionPreparation {
    std::shared_ptr<const moe::Plan> experts;
};

class Startup {
public:
    using Types         = RuntimeTypes;
    using ExecutionPlan = models::ExecutionPlan<qwen4_exp::ExecutionPlan, ExecutionPreparation>;
    using Calibration   = GraphCalibration;

    Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
            const EngineOptions&, Device);

    [[nodiscard]] Architecture architecture() const noexcept { return Architecture::Qwen4Exp; }

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
    std::shared_ptr<const ple::disk::Table> table_;
};

[[nodiscard]] qwen::Frontend create_frontend(const ModelDefinition&, const EngineOptions&);

[[nodiscard]] std::unique_ptr<RankProgram>
create_rank_program(const execution::Parameters&, ExecutionPlan&&, DeviceContext&,
                    const StartupObserver&, GraphPreparationMode = GraphPreparationMode::Complete);
[[nodiscard]] std::string prefill_signature(const ModelDefinition&, const moe::Plan*,
                                            std::uint32_t chunk);

} // namespace ninfer::models::qwen4_exp
