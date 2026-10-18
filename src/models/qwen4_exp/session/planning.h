#pragma once

#include "models/common/context/directory.h"
#include "models/qwen4_exp/execution_shape.h"

namespace ninfer::models::qwen4_exp {

struct ExecutionPlan;

namespace execution {
struct ModelProfile;
}

struct SessionPlan {
    std::shared_ptr<const execution::ModelProfile> profile;
    ExecutionShape shape;
    context::ContextDirectorySpec context;
    std::size_t vision_max_merged_tokens = 0;
};

[[nodiscard]] SessionPlan plan_session(const ExecutionPlan&);

} // namespace ninfer::models::qwen4_exp
