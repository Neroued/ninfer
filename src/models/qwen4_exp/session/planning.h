#pragma once

#include "models/common/context/directory.h"
#include "models/qwen4_exp/execution_shape.h"

namespace ninfer::models::qwen4_exp {

struct ExecutionPlan;

struct SessionPlan {
    ExecutionShape shape;
    context::ContextDirectorySpec context;
    std::size_t vision_max_merged_tokens = 0;
    std::uint32_t checkpoint_interval    = 8192;

    bool operator==(const SessionPlan&) const = default;
};

[[nodiscard]] SessionPlan plan_session(const ExecutionPlan&);
[[nodiscard]] std::size_t session_metadata_bytes(const ExecutionPlan&);

} // namespace ninfer::models::qwen4_exp
