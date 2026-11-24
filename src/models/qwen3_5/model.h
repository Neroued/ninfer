#pragma once

#include "models/qwen3_5/definition.h"

namespace ninfer::models::qwen3_5 {

class ResolvedLoadPlan;

// Local immutable weight storage. Borrowers finish before its backing is released.
class ModelShard {
public:
    ~ModelShard();
    ModelShard(const ModelShard&)            = delete;
    ModelShard& operator=(const ModelShard&) = delete;

    [[nodiscard]] const ModelDefinition& definition() const { return *definition_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightId id) const { return weight_input(bound_, id); }

    [[nodiscard]] ops::WeightInput input(WeightUseId id) const { return weight_input(bound_, id); }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const { return bound_; }

    [[nodiscard]] std::span<const ExpertBankSource> expert_banks() const { return banks_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                         const StartupObserver*);
    ModelShard(std::shared_ptr<const ModelDefinition>, artifact::MaterializedArtifact,
               std::vector<BoundWeight>, std::vector<ExpertBankSource>);

    std::shared_ptr<const ModelDefinition> definition_;
    artifact::MaterializedArtifact backing_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBankSource> banks_;
};

} // namespace ninfer::models::qwen3_5
