#pragma once

#include "models/common/moe/schedule.h"
#include "ninfer/types.h"

namespace ninfer::models::qwen3_5 {
class LoadPlan;
struct TextConfig;

namespace execution {
struct ModelProfile;
}

namespace detail {

// Empty when the model geometry has no calibrated default.
[[nodiscard]] std::span<const std::uint32_t> default_expert_priority(const TextConfig&,
                                                                     std::size_t layer);
[[nodiscard]] std::shared_ptr<const moe::Plan>
describe_moe_plan(const LoadPlan&, const execution::ModelProfile&, const MoeOptions& = {});

} // namespace detail
} // namespace ninfer::models::qwen3_5
