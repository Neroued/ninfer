#pragma once

#include "models/common/context/directory.h"
#include "models/common/speculative/round_plan.h"
#include "models/inkling2/execution/workspace.h"
#include "models/inkling2/program/frame.h"
#include "models/inkling2/state/context.h"
#include "runtime/engine/context_capacity.h"

#include <cstddef>
#include <memory>

namespace ninfer::models::inkling2 {

struct ExecutionPlan {
    std::shared_ptr<const execution::ModelProfile> profile;
    ExecutionShape shape;
    std::shared_ptr<const BatchExecutionPlan> batches;
    std::shared_ptr<const speculative::DecodeGraphRoutes> decode_routes;
    StateLayout state;
    CacheLayout main;
    std::optional<CacheLayout> dspark;
    FrameLayout frame;
    execution::WorkspacePlan workspace;
    std::size_t fixed_bytes         = 0;
    std::size_t mapping_elements    = 0;
    std::size_t pinned_output_bytes = 0;
    runtime::ContextCapacityResolution capacity;
    bool use_cuda_graph = false;

    [[nodiscard]] std::size_t pinned_bytes() const;
    [[nodiscard]] std::size_t pinned_context_counts_bytes() const;
    [[nodiscard]] std::size_t fixed_device_bytes() const;
    [[nodiscard]] std::size_t device_bytes() const;
    [[nodiscard]] std::size_t reservation_bytes() const;
};

[[nodiscard]] ExecutionPlan plan_execution(std::shared_ptr<const execution::ModelProfile>,
                                           const EngineOptions&, std::int32_t sm_count,
                                           BatchExecutionSpec);
[[nodiscard]] runtime::ContextCapacityPlan describe_execution_capacity(const ExecutionPlan&);
[[nodiscard]] context::ContextDirectorySpec describe_context(const ExecutionPlan&);

} // namespace ninfer::models::inkling2
