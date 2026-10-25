#pragma once
#include "models/glm5next/execution_shape.h"
#include "models/glm5next/state/context.h"
#include "models/common/context/directory.h"

namespace ninfer::models::glm5next {
struct SessionPlan {
    ExecutionShape shape;
    context::ContextDirectorySpec context;
    std::vector<StatePlacement> states;
    std::vector<KVPageLayout> caches;
    std::uint32_t checkpoint_interval = 8192;
};

[[nodiscard]] runtime::ContextMetadataPlan
session_metadata_plan(const context::ContextDirectorySpec&, const ContextPoolGeometry&,
                      std::uint32_t rank);
} // namespace ninfer::models::glm5next
