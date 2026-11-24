#pragma once

#include "models/common/context/directory.h"
#include "models/glm5next/execution/workspace.h"
#include "models/glm5next/program/frame.h"
#include "models/glm5next/state/context.h"
#include "runtime/engine/context_capacity.h"

namespace ninfer::models::glm5next {

struct ExecutionPlan {
    std::shared_ptr<const execution::ModelProfile> profile;
    ExecutionShape shape;
    StateLayout state;
    CacheLayout main;
    std::optional<CacheLayout> mtp;
    FrameLayout frame;
    execution::WorkspacePlan workspace;
    std::array<LayoutRegion, 2> peer_staging;
    std::size_t fixed_bytes         = 0;
    std::size_t mapping_elements    = 0;
    std::size_t pinned_output_bytes = 0;
    runtime::ContextCapacityResolution capacity;
    bool use_cuda_graph = false;

    [[nodiscard]] std::size_t pinned_bytes() const;
    [[nodiscard]] std::size_t device_bytes() const;
    [[nodiscard]] std::size_t reservation_bytes() const;
};

// Plans eager execution resources. The Program's graph preparation accounts its own storage.
[[nodiscard]] ExecutionPlan plan_execution(std::shared_ptr<const execution::ModelProfile>,
                                           const EngineOptions&, KvCacheVectorFormat index_format,
                                           std::int32_t sm_count);
[[nodiscard]] runtime::ContextCapacityPlan describe_execution_capacity(const ExecutionPlan&);
// Local parts only; the group preparation assembles the complete directory before construction.
[[nodiscard]] context::ContextDirectorySpec describe_context(const ExecutionPlan&);

} // namespace ninfer::models::glm5next
