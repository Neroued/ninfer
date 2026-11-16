#pragma once

#include "models/common/context/directory.h"
#include "models/qwen4_exp/execution_shape.h"
#include "models/common/batch.h"

namespace ninfer::models::qwen4_exp {

struct ExecutionPlan;

struct SessionPlan {
    ExecutionShape shape;
    std::shared_ptr<const BatchExecutionPlan> batches;
    context::ContextDirectorySpec context;
    std::uint32_t vision_max_features = 0;
    std::uint32_t checkpoint_interval = 8192;

    bool operator==(const SessionPlan& other) const {
        return shape == other.shape && context == other.context &&
               vision_max_features == other.vision_max_features &&
               checkpoint_interval == other.checkpoint_interval && batches && other.batches &&
               batches->spec() == other.batches->spec();
    }
};

[[nodiscard]] SessionPlan plan_session(const ExecutionPlan&);
[[nodiscard]] runtime::ContextMetadataPlan session_metadata_plan(const ExecutionPlan&,
                                                                 const ContextPoolGeometry&);

} // namespace ninfer::models::qwen4_exp
