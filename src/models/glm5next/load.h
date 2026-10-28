#pragma once

#include "models/common/parallel_weights.h"
#include "models/glm5next/model.h"

namespace ninfer::models::glm5next {

// Borrows the Reader until materialization finishes. Definitions and shards own everything
// needed after that point; planning does not allocate device storage.
class ResolvedLoadPlan {
public:
    ResolvedLoadPlan(ResolvedLoadPlan&&) noexcept            = default;
    ResolvedLoadPlan& operator=(ResolvedLoadPlan&&) noexcept = default;
    ResolvedLoadPlan(const ResolvedLoadPlan&)                = delete;
    ResolvedLoadPlan& operator=(const ResolvedLoadPlan&)     = delete;

    [[nodiscard]] const ModelDefinition& definition() const { return *definition_; }

    [[nodiscard]] const RankPlan& rank_plan() const { return rank_; }

    [[nodiscard]] std::span<const moe::ExpertDescription> expert_data() const {
        return selection_.expert_descriptions;
    }

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const {
        return materialization_;
    }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const { return descriptions_; }

private:
    friend ResolvedLoadPlan resolve_rank_load(const artifact::Reader&,
                                              std::shared_ptr<const ModelDefinition>, RankPlan,
                                              Device);
    friend std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                         const StartupObserver*);
    ResolvedLoadPlan(const artifact::Reader&, std::shared_ptr<const ModelDefinition>, RankPlan,
                     SelectedWeights, artifact::MaterializationPlan, Device);

    const artifact::Reader* reader_;
    std::shared_ptr<const ModelDefinition> definition_;
    RankPlan rank_;
    SelectedWeights selection_;
    artifact::MaterializationPlan materialization_;
    Device device_;
    std::vector<WeightParent> parents_;
    std::vector<BoundWeight> descriptions_;
};

[[nodiscard]] ResolvedLoadPlan resolve_rank_load(const artifact::Reader&,
                                                 std::shared_ptr<const ModelDefinition>, RankPlan,
                                                 Device);
[[nodiscard]] std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                            const StartupObserver* = nullptr);

} // namespace ninfer::models::glm5next
