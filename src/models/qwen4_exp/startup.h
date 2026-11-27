#pragma once

#include "models/common/startup.h"
#include "runtime/engine/placement_selection.h"
#include "models/qwen4_exp/load.h"
#include "models/qwen4_exp/runtime_types.h"
#include "models/qwen4_exp/program/graph_preparation.h"
#include "models/registry.h"

namespace ninfer::models::qwen4_exp {

struct ExecutionPreparation {
    std::shared_ptr<const moe::Plan> experts;
};

struct LocalPlacement {
    WeightPlacement weights;
    ple::Placement table;
};

class Startup {
public:
    using Types         = RuntimeTypes;
    using ExecutionPlan =
        models::ExecutionPlan<qwen4_exp::ExecutionPlan, ExecutionPreparation, LocalPlacement>;
    using Calibration   = GraphCalibration;

    Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
            const EngineOptions&, Device);

    [[nodiscard]] Architecture architecture() const noexcept { return Architecture::Qwen4Exp; }

    [[nodiscard]] std::optional<ExecutionPlan> plan(DeviceContext&, const EngineOptions&,
                                                    LocalPlacement, std::size_t available,
                                                    std::size_t available_host_bytes) const;
    [[nodiscard]] std::unique_ptr<ModelShard> materialize(DeviceContext&, LocalPlacement,
                                                          const StartupObserver&);
    [[nodiscard]] qwen::Frontend frontend(const EngineOptions&) const;
    [[nodiscard]] GraphPreparationPlan graph_plan(const EngineOptions&,
                                                  const ExecutionPreparation&) const;
    [[nodiscard]] Calibration calibrate(const execution::Parameters&, DeviceContext&,
                                        const EngineOptions&, const ExecutionPreparation&) const;

    struct Candidate {
        LocalPlacement placement;
        std::size_t device_bytes, host_bytes;
        std::size_t preference, upper_blocks;
    };

    [[nodiscard]] std::vector<Candidate> placements(DeviceContext&, const EngineOptions&,
                                                    std::size_t available, std::size_t host,
                                                    bool shared) const;

    [[nodiscard]] static runtime::PlacementEstimate estimate(std::size_t id, const Candidate& c) {
        return {id, c.placement.weights == WeightPlacement::Resident ? 0U : 1U, c.preference,
                c.upper_blocks};
    }

    [[nodiscard]] static std::string placement_name(const Candidate& c) {
        return ple::name(c.placement.table);
    }

    void report_placement(std::span<const Candidate>, std::size_t, std::size_t measured_blocks,
                          std::string_view reasons) const;

    [[nodiscard]] const artifact::MaterializationStats& last_load() const { return last_load_; }

    void retain_weights(std::unique_ptr<ModelShard>);
    void release_other_weights(LocalPlacement);
    [[nodiscard]] Candidate outstanding(Candidate) const;
    [[nodiscard]] std::size_t capacity_blocks(const ExecutionPlan&, const EngineOptions&,
                                              std::size_t offered, std::size_t host) const;

    [[nodiscard]] std::string prefill_signature(const ExecutionPreparation&,
                                                std::uint32_t chunk) const;

    [[nodiscard]] const std::shared_ptr<const ModelDefinition>& definition() const {
        return definition_;
    }

    std::shared_ptr<const execution::ModelProfile> profile;
private:
    std::shared_ptr<const BatchExecutionPlan> batches_;

    struct TableChoice {
        ple::Placement placement;
        ple::disk::TableGeometry geometry;
        std::size_t device_bytes, host_bytes;
        std::shared_ptr<const execution::ModelProfile> profile;
    };

    const TableChoice& table(ple::Placement) const;
    std::vector<TableChoice> tables_;
    std::vector<WeightCandidate> weights_;
    std::unique_ptr<ModelShard> retained_;
    artifact::MaterializationStats last_load_;
    std::optional<WeightPlacement> current_weights_;
    std::shared_ptr<const moe::Plan> offload_;
    const artifact::Reader& reader_;
    std::shared_ptr<const ModelDefinition> definition_;
};

[[nodiscard]] qwen::Frontend create_frontend(const ModelDefinition&, const EngineOptions&);

[[nodiscard]] std::unique_ptr<RankProgram>
create_rank_program(const execution::Parameters&, ExecutionPlan&&, DeviceContext&,
                    const StartupObserver&, GraphPreparationMode = GraphPreparationMode::Complete);
[[nodiscard]] std::string prefill_signature(const ModelDefinition&, const moe::Plan*,
                                            std::uint32_t chunk);

} // namespace ninfer::models::qwen4_exp
