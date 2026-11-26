#pragma once

#include "distributed/topology.h"
#include "models/common/parallel.h"
#include "models/inkling2/definition.h"

namespace ninfer::models::inkling2 {

struct RankPlan {
    std::uint32_t rank = 0, size = 1, ep_size = 1;
    IndexRange vocabulary;
    std::vector<ParameterAssembly> assemblies;
    std::vector<ExpertPartition> experts;
};

struct ParallelPlan {
    distributed::ExecutionTopology topology;
    std::vector<RankPlan> ranks;
};

[[nodiscard]] RankPlan plan_single_rank(const ModelDefinition&);
[[nodiscard]] ParallelPlan plan_parallel(const ModelDefinition&, const DeploymentOptions&);

} // namespace ninfer::models::inkling2
