#pragma once

#include "models/qwen4_exp/model.h"
#include "models/qwen4_exp/parallel/weights.h"

namespace ninfer::models::qwen4_exp {

// One local materialization. The Reader remains alive until materialize_model returns;
// the definition and all resulting storage survive independently of it.
class ResolvedLoadPlan {
public:
    ResolvedLoadPlan(ResolvedLoadPlan&&) noexcept            = default;
    ResolvedLoadPlan& operator=(ResolvedLoadPlan&&) noexcept = default;
    ResolvedLoadPlan(const ResolvedLoadPlan&)                = delete;
    ResolvedLoadPlan& operator=(const ResolvedLoadPlan&)     = delete;

    [[nodiscard]] const ModelDefinition& definition() const { return *definition_; }

    [[nodiscard]] const RankPlan& rank_plan() const { return rank_; }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const {
        return descriptions_.empty() ? definition_->weight_data() : descriptions_;
    }

    [[nodiscard]] std::span<const moe::ExpertDescription> expert_data() const {
        return selection_ ? std::span<const moe::ExpertDescription>(
                                selection_->weights.expert_descriptions)
                          : std::span<const moe::ExpertDescription>{};
    }

    [[nodiscard]] const artifact::MaterializationPlan& materialization() const {
        return materialization_;
    }

    [[nodiscard]] const ple::disk::Table& ple_table() const { return *ple_; }

    [[nodiscard]] const std::shared_ptr<const ple::disk::Table>& ple_table_owner() const {
        return ple_;
    }

private:
    friend ResolvedLoadPlan resolve_load(const artifact::Reader&,
                                         std::shared_ptr<const ModelDefinition>, Device,
                                         std::optional<Device>,
                                         std::shared_ptr<const ple::disk::Table>);
    friend ResolvedLoadPlan resolve_rank_load(const artifact::Reader&,
                                              std::shared_ptr<const ModelDefinition>, RankPlan,
                                              Device);
    friend std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                         const StartupObserver*);
    ResolvedLoadPlan(const artifact::Reader&, std::shared_ptr<const ModelDefinition>,
                     artifact::MaterializationPlan, std::vector<Device>, std::vector<Device>,
                     std::shared_ptr<const ple::disk::Table>);
    const artifact::Reader* reader_;
    std::shared_ptr<const ModelDefinition> definition_;
    artifact::MaterializationPlan materialization_;
    std::vector<Device> devices_, bank_devices_;
    std::shared_ptr<const ple::disk::Table> ple_;
    RankPlan rank_;
    std::optional<WeightSelection> selection_;
    std::vector<WeightParent> parents_;
    std::vector<BoundWeight> descriptions_;
};

[[nodiscard]] ResolvedLoadPlan resolve_load(const artifact::Reader&,
                                            std::shared_ptr<const ModelDefinition>, Device,
                                            std::optional<Device> main_experts            = {},
                                            std::shared_ptr<const ple::disk::Table> table = {});
[[nodiscard]] std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                            const StartupObserver* = nullptr);
[[nodiscard]] ResolvedLoadPlan resolve_rank_load(const artifact::Reader&,
                                                 std::shared_ptr<const ModelDefinition>, RankPlan,
                                                 Device);

} // namespace ninfer::models::qwen4_exp
