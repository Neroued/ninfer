#pragma once

#include "distributed/topology.h"
#include "models/common/moe/placement.h"
#include "models/glm5next/definition.h"

namespace ninfer::models::glm5next {

struct DraftPartition {
    IndexRange features, query_heads, key_value_heads, intermediate;
};

struct RankPlan {
    RankPlacement placement;
    IndexRange layers, banks;

    [[nodiscard]] ExpertBankId local_bank(ExpertBankId global) const {
        if (global.index < banks.begin || global.index - banks.begin >= banks.size)
            throw std::out_of_range("GLM5Next expert bank belongs to another pipeline stage");
        return {global.index - banks.begin};
    }

    IndexRange mla_heads;
    IndexRange kda_heads, vocabulary, dense_intermediate, shared_intermediate;
    std::optional<DraftPartition> draft;
    std::vector<WeightId> whole;
    std::vector<ParameterAssembly> assemblies;
    std::vector<ExpertPartition> experts;
    std::shared_ptr<const moe::ExpertDomain> expert_domain;
    std::vector<WeightId> omitted;
};

struct ParallelPlan {
    distributed::ExecutionTopology topology;
    std::vector<RankPlan> ranks;
};

[[nodiscard]] std::vector<PipelineLayerDemand> pipeline_layer_demands(const ModelDefinition&);

[[nodiscard]] std::uint64_t compute_rank_shard_digest(const artifact::ArtifactId&, LoadOptions,
                                                      const RankPlan&);
[[nodiscard]] RankPlan plan_rank_layers(const ModelDefinition&, const distributed::RankIdentity&,
                                        IndexRange);
[[nodiscard]] RankPlan plan_single_rank(const ModelDefinition&);
[[nodiscard]] ParallelPlan plan_parallel(const ModelDefinition&, const DeploymentOptions&);

} // namespace ninfer::models::glm5next
