#pragma once

#include "distributed/topology.h"
#include "models/common/parallel.h"
#include "models/qwen3_5/definition.h"

#include <cstdint>
#include <vector>

namespace ninfer::models::qwen3_5 {

struct RankPlan {
    std::uint32_t rank    = 0;
    std::uint32_t size    = 1;
    std::uint32_t ep_size = 1;
    std::vector<WeightId> whole;
    std::vector<ParameterAssembly> assemblies;
    std::vector<ExpertPartition> experts;
    std::vector<WeightId> omitted;
};

struct ParallelPlan {
    distributed::ExecutionTopology topology;
    std::vector<RankPlan> ranks;
};

[[nodiscard]] std::uint64_t compute_rank_shard_digest(const artifact::ArtifactId&, LoadOptions,
                                                      const RankPlan&);
[[nodiscard]] RankPlan plan_single_rank(const ModelDefinition&);
[[nodiscard]] ParallelPlan plan_parallel(const ModelDefinition&, const DeploymentOptions&);

} // namespace ninfer::models::qwen3_5
