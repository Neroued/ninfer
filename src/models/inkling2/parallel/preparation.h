#pragma once

#include "models/common/batch.h"
#include "models/common/context/directory.h"
#include "models/inkling2/execution_shape.h"
#include "models/inkling2/parallel/plan.h"
#include "models/inkling2/state/context.h"
#include "ninfer/observability/resources.h"

#include <memory>
#include <span>
#include <vector>

namespace ninfer::models::inkling2 {

struct RankResourceDescription {
    std::uint32_t rank = 0;
    ExecutionShape shape;
    BatchExecutionSpec batches;
    StatePlacement state;
    context::ContextDirectorySpec context;
    RankMemoryBudget budget;
    RankSnapshot observation;
    std::vector<KVPageGeometry> caches;
};

struct RankLoadRequest {
    artifact::ArtifactId artifact;
    RankPlan rank;
    EngineOptions options;
    BatchExecutionSpec batches;
    std::size_t host_reservation_bytes = 0;
    bool control_owner                 = true;
};

struct PreparedContext {
    ExecutionShape shape;
    std::shared_ptr<const BatchExecutionPlan> batches;
    std::vector<StatePlacement> states;
    std::unique_ptr<context::ContextDirectory> directory;
    context::ContextDirectorySpec spec;
    std::vector<KVPageGeometry> caches;
};

[[nodiscard]] context::ContextDirectorySpec
plan_group_context(const ParallelPlan&, std::span<const RankResourceDescription>);

[[nodiscard]] PreparedContext prepare_context(const ParallelPlan&,
                                              std::span<const RankResourceDescription>);

} // namespace ninfer::models::inkling2
