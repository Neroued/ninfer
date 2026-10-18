#pragma once

#include "models/qwen4_exp/definition.h"
#include "models/qwen4_exp/ple/disk/table.h"

namespace ninfer::models::qwen4_exp {

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

    [[nodiscard]] std::span<const BoundWeight> weight_data() const { return bound_; }

    [[nodiscard]] std::span<const ExpertBankSource> expert_banks() const { return banks_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const {
        return backing_.stats();
    }

    [[nodiscard]] const ple::disk::Table& ple_table() const { return *ple_; }

private:
    friend std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                         const StartupObserver*);
    ModelShard(std::shared_ptr<const ModelDefinition>, artifact::MaterializedArtifact,
               std::vector<BoundWeight>, std::vector<ExpertBankSource>,
               std::shared_ptr<const ple::disk::Table>);

    std::shared_ptr<const ModelDefinition> definition_;
    artifact::MaterializedArtifact backing_;
    std::shared_ptr<const ple::disk::Table> ple_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBankSource> banks_;
};

} // namespace ninfer::models::qwen4_exp
