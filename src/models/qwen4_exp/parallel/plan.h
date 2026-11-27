#pragma once

#include "models/common/moe/placement.h"
#include "models/qwen4_exp/definition.h"
#include "distributed/topology.h"

namespace ninfer::models::qwen4_exp {

struct RankPlan {
    RankPlacement placement;
    ple::Placement ple_placement = ple::Placement::Disk;
    std::vector<ple::Placement> ple_candidates{ple::Placement::Disk};
    IndexRange gdn_keys, gdn_values, qsa_queries, qsa_kv, vocabulary, ple_shards;
    std::uint32_t shared_intermediate = 0;
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

[[nodiscard]] std::uint64_t compute_rank_shard_digest(const artifact::ArtifactId&, LoadOptions,
                                                      const RankPlan&);
[[nodiscard]] ParallelPlan plan_parallel(const ModelDefinition&, const DeploymentOptions&,
                                         ple::Placement);
[[nodiscard]] ParallelPlan plan_parallel(const ModelDefinition&, const DeploymentOptions&,
                                         std::span<const ple::Placement>);
[[nodiscard]] RankPlan plan_single_rank(const ModelDefinition&, ple::Placement = ple::Placement::Disk);
[[nodiscard]] std::span<const std::uint32_t> default_expert_priority(const TextConfig&,
                                                                     std::size_t layer);

} // namespace ninfer::models::qwen4_exp
