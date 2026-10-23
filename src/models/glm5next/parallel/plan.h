#pragma once

#include "distributed/topology.h"
#include "models/common/parallel.h"
#include "models/glm5next/definition.h"

namespace ninfer::models::glm5next {

struct RankPlan {
    std::uint32_t rank = 0, size = 1;
    IndexRange kda_heads, vocabulary, dense_intermediate, shared_intermediate;
    std::vector<ParameterAssembly> assemblies;
    std::vector<std::vector<std::uint32_t>> experts;
};

struct ParallelPlan {
    distributed::ExecutionTopology topology;
    std::vector<RankPlan> ranks;
};

[[nodiscard]] RankPlan plan_single_rank(const ModelDefinition&);
[[nodiscard]] ParallelPlan plan_parallel(const ModelDefinition&, const DeploymentOptions&);

} // namespace ninfer::models::glm5next
