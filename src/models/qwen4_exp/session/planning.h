#pragma once

#include "models/common/context/directory.h"
#include "models/qwen4_exp/execution_shape.h"
#include "models/common/batch.h"
#include "models/common/speculative/round_plan.h"

namespace ninfer::models::qwen4_exp {

struct ExecutionPlan;

struct SessionPlan {
    ExecutionShape shape;
    std::shared_ptr<const BatchExecutionPlan> batches;
    context::ContextDirectorySpec context;
    std::uint32_t vision_max_features = 0;
    std::uint32_t checkpoint_interval = 8192;

    std::shared_ptr<const speculative::DecodeGraphRoutes> decode_routes;
    ExecutionPath decode_path = ExecutionPath::Eager;

    bool operator==(const SessionPlan& other) const {
        return decode_path == other.decode_path && shape == other.shape &&
               context == other.context && vision_max_features == other.vision_max_features &&
               checkpoint_interval == other.checkpoint_interval && batches && other.batches &&
               batches->spec() == other.batches->spec();
    }
};

[[nodiscard]] SessionPlan plan_session(const ExecutionPlan&);
[[nodiscard]] runtime::ContextMetadataPlan session_metadata_plan(const ExecutionPlan&,
                                                                 const ContextPoolGeometry&);

} // namespace ninfer::models::qwen4_exp
