#pragma once

#include "models/glm5next/program/planning.h"

namespace ninfer::models::glm5next {

struct RankResourceDescription {
    std::uint32_t rank;
    ExecutionShape shape;
    StateLayout state;
    context::ContextDirectorySpec context;
    RankMemoryUsage memory;
    RankMemoryBudget budget;
};

// Combines local record specifications; capacities remain independent. Construction of the
// resulting directory is owned by the coordinator and precedes execution activation.
[[nodiscard]] context::ContextDirectorySpec
plan_group_context(const ParallelPlan&, std::span<const RankResourceDescription>);

} // namespace ninfer::models::glm5next
