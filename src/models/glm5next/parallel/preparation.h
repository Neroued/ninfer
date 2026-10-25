#pragma once

#include "models/common/context/directory.h"
#include "models/glm5next/execution_shape.h"
#include "models/glm5next/parallel/plan.h"
#include "models/glm5next/state/context.h"
#include "ninfer/observability/resources.h"

namespace ninfer::models::glm5next {

struct RankResourceDescription {
    std::uint32_t rank;
    ExecutionShape shape;
    StatePlacement state;
    context::ContextDirectorySpec context;
    RankMemoryBudget budget;
    RankSnapshot observation;
    std::uint32_t vision_max_features = 0;
    std::vector<KVPageGeometry> caches;
};

struct RankLoadRequest {
    artifact::ArtifactId artifact;
    RankPlan rank;
    EngineOptions options;
    std::size_t host_reservation_bytes = 0;
    bool control_owner                 = false;
};

struct PreparedContext {
    ExecutionShape shape;
    std::vector<StatePlacement> states;
    std::uint32_t vision_max_features;
    std::unique_ptr<context::ContextDirectory> directory;
    context::ContextDirectorySpec spec;
    std::vector<KVPageGeometry> caches;
};

// Combines local record specifications; capacities remain independent. Construction of the
// resulting directory is owned by the coordinator and precedes execution activation.
[[nodiscard]] context::ContextDirectorySpec
plan_group_context(const ParallelPlan&, std::span<const RankResourceDescription>);

[[nodiscard]] PreparedContext prepare_context(const ParallelPlan&,
                                              std::span<const RankResourceDescription>);

} // namespace ninfer::models::glm5next
