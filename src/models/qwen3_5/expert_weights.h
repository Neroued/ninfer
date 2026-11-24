#pragma once

#include "models/common/moe/weights.h"
#include "models/qwen3_5/weights.h"
#include "models/common/bindings.h"

namespace ninfer::models::qwen3_5 {

[[nodiscard]] std::vector<moe::BankDescription>
describe_expert_weights(const artifact::Reader&, ModelWeights&, std::span<const PendingWeight>);

} // namespace ninfer::models::qwen3_5
