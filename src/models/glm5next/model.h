#pragma once

#include "models/glm5next/parallel/plan.h"

namespace ninfer::models::glm5next {

class ResolvedLoadPlan;

// Owns one rank's immutable selected weights and expert metadata.
class ModelShard {
public:
    ModelShard(const ModelShard&)            = delete;
    ModelShard& operator=(const ModelShard&) = delete;

    [[nodiscard]] const ModelDefinition& definition() const { return *definition_; }

    [[nodiscard]] const RankPlan& rank_plan() const { return rank_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return weights_.at(id.index); }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const { return weights_; }

    [[nodiscard]] ops::WeightInput input(WeightId id) const { return weight_input(weights_, id); }

    [[nodiscard]] const moe::ResidentExpertBank& experts(ExpertBankId id) const {
        return experts_.at(id.index);
    }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const {
        return backing_.stats();
    }

    [[nodiscard]] std::size_t metadata_bytes() const;

private:
    friend std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                         const StartupObserver*);
    ModelShard(std::shared_ptr<const ModelDefinition>, RankPlan, artifact::MaterializedArtifact);

    std::shared_ptr<const ModelDefinition> definition_;
    RankPlan rank_;
    artifact::MaterializedArtifact backing_;
    std::vector<WeightParent> parents_;
    std::vector<BoundWeight> weights_;
    std::vector<moe::ResidentExpertBank> experts_;
};

} // namespace ninfer::models::glm5next
