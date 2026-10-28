#pragma once

#include "models/qwen4_exp/definition.h"
#include "models/qwen4_exp/ple/disk/table.h"
#include "models/qwen4_exp/parallel/plan.h"
#include "ninfer/ops/ple.h"

namespace ninfer::models::qwen4_exp {

class ResolvedLoadPlan;

struct DevicePleTable {
    DeviceBuffer decode;
    std::vector<ops::Nf4EmbeddingShard> shards;
};

using PleTable = std::variant<std::shared_ptr<const ple::disk::Table>, DevicePleTable>;

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

    [[nodiscard]] std::span<const moe::ExpertDescription> expert_data() const {
        return expert_descriptions_;
    }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const {
        return backing_.stats();
    }

    [[nodiscard]] const PleTable& ple_table() const { return ple_; }

    [[nodiscard]] const RankPlan& rank_plan() const { return rank_; }

    [[nodiscard]] const moe::ResidentExpertBank* resident_experts(ExpertBankId id) const {
        return local_banks_.empty() ? nullptr : &local_banks_.at(id.index);
    }

    [[nodiscard]] std::size_t metadata_bytes() const;

private:
    friend std::unique_ptr<ModelShard> materialize_model(ResolvedLoadPlan&&, DeviceContext&,
                                                         const StartupObserver*);
    ModelShard(std::shared_ptr<const ModelDefinition>, artifact::MaterializedArtifact,
               std::vector<BoundWeight>, std::vector<ExpertBankSource>,
               std::shared_ptr<const ple::disk::Table>);

    std::shared_ptr<const ModelDefinition> definition_;
    artifact::MaterializedArtifact backing_;
    PleTable ple_;
    std::vector<BoundWeight> bound_;
    std::vector<ExpertBankSource> banks_;
    RankPlan rank_;
    std::vector<moe::ExpertDescription> expert_descriptions_;
    std::vector<WeightParent> local_parents_;
    std::vector<moe::ResidentExpertBank> local_banks_;
};

} // namespace ninfer::models::qwen4_exp
