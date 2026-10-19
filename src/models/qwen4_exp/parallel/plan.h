#pragma once

#include "models/common/parallel.h"
#include "models/qwen4_exp/definition.h"
#include "distributed/topology.h"

namespace ninfer::models::qwen4_exp {

struct RankPlan {
    std::uint32_t rank = 0, size = 1;
    ple::Placement ple_placement = ple::Placement::Disk;
    IndexRange gdn_keys, gdn_values, qsa_queries, qsa_kv, vocabulary, ple_shards;
    std::uint32_t shared_intermediate = 0;
    std::vector<ParameterAssembly> assemblies;
    std::vector<std::vector<std::uint32_t>> experts;
};

struct ParallelPlan {
    distributed::ExecutionTopology topology;
    std::vector<RankPlan> ranks;
};

[[nodiscard]] ParallelPlan plan_parallel(const ModelDefinition&, const DeploymentOptions&);
[[nodiscard]] RankPlan plan_single_rank(const ModelDefinition&);
[[nodiscard]] std::span<const std::uint32_t> default_expert_priority(const TextConfig&,
                                                                     std::size_t layer);

} // namespace ninfer::models::qwen4_exp
