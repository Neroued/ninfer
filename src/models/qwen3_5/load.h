#pragma once

#include "models/qwen3_5/model.h"
#include "models/qwen3_5/parallel/plan.h"
#include "models/common/parallel_weights.h"

namespace ninfer::models::qwen3_5 {

// One local materialization. The Reader remains alive until materialize_model returns;
// the definition and all resulting storage survive independently of it.
class ResolvedLoadPlan {
public:
    ResolvedLoadPlan(ResolvedLoadPlan&&) noexcept            = default;
    ResolvedLoadPlan& operator=(ResolvedLoadPlan&&) noexcept = default;
    ResolvedLoadPlan(const ResolvedLoadPlan&)                = delete;
    ResolvedLoadPlan& operator=(const ResolvedLoadPlan&)     = delete;

    [[nodiscard]] const ModelDefinition& definition() const { return *definition_; }

    [[nodiscard]] std::span<const moe::ExpertDescription> expert_data() const {
        return selection_ ? std::span<const moe::ExpertDescription>(selection_->expert_descriptions)
                          : std::span<const moe::ExpertDescription>{};
    }

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const {
        return materialization_;
    }

private:
    friend ResolvedLoadPlan resolve_load(const artifact::Reader&,
                                         std::shared_ptr<const ModelDefinition>, Device,
                                         std::optional<Device>);
    friend ResolvedLoadPlan resolve_rank_load(const artifact::Reader&,
                                              std::shared_ptr<const ModelDefinition>,
                                              const RankPlan&, Device);
    friend std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                         const StartupObserver*);
    ResolvedLoadPlan(const artifact::Reader&, std::shared_ptr<const ModelDefinition>,
                     artifact::MaterializationPlan, std::vector<Device>, std::vector<Device>);
    const artifact::Reader* reader_;
    std::shared_ptr<const ModelDefinition> definition_;
    artifact::MaterializationPlan materialization_;
    std::vector<Device> devices_, bank_devices_;
    std::optional<SelectedWeights> selection_;
};

[[nodiscard]] ResolvedLoadPlan resolve_load(const artifact::Reader&,
                                            std::shared_ptr<const ModelDefinition>, Device,
                                            std::optional<Device> main_experts = {});
[[nodiscard]] ResolvedLoadPlan resolve_rank_load(const artifact::Reader&,
                                                 std::shared_ptr<const ModelDefinition>,
                                                 const RankPlan&, Device);
[[nodiscard]] std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                            const StartupObserver* = nullptr);

} // namespace ninfer::models::qwen3_5
