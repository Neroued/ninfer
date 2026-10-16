#pragma once

#include "models/common/startup.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/program/runtime_types.h"
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
    static constexpr bool supports_graph = true;

    Startup(const artifact::Reader&, const EngineOptions&, Device);

    [[nodiscard]] Architecture architecture() const noexcept { return Architecture::Qwen4Exp; }

    [[nodiscard]] std::optional<ExecutionPlan> plan(DeviceContext&, const EngineOptions&,
                                                    WeightPlacement, std::size_t available) const;
    [[nodiscard]] std::unique_ptr<Model> materialize(DeviceContext&, WeightPlacement,
                                                     const StartupObserver&);
    [[nodiscard]] qwen::Frontend frontend(const Model&, const EngineOptions&) const;
    [[nodiscard]] GraphPreparationPlan graph_plan(const EngineOptions&,
                                                  const ExecutionPreparation&) const;
    [[nodiscard]] Calibration calibrate(const execution::Parameters&, DeviceContext&,
                                        const EngineOptions&, const ExecutionPreparation&) const;

    [[nodiscard]] std::span<const WeightCandidate> placements() const { return placements_; }

    [[nodiscard]] std::string prefill_signature(const Model&, const ExecutionPreparation&,
                                                std::uint32_t chunk) const;
    std::shared_ptr<const execution::ModelProfile> profile;
private:
    std::vector<WeightCandidate> placements_;
    std::shared_ptr<const moe::Plan> offload_;
    const artifact::Reader& reader_;
    LoadOptions selection_;
    std::optional<LoadPlan> source_;
    ple::disk::TableGeometry table_;
};

[[nodiscard]] std::unique_ptr<Program> create_program(const execution::Parameters&, ExecutionPlan&&,
                                                      DeviceContext&, const StartupObserver&);
[[nodiscard]] std::string prefill_signature(const Model&, const moe::Plan*, std::uint32_t chunk);

} // namespace ninfer::models::qwen4_exp
