#pragma once

#include "models/common/parallel_weights.h"
#include "models/qwen4_exp/parallel/plan.h"

namespace ninfer::models::qwen4_exp {

[[nodiscard]] SelectedWeights select_weights(const artifact::Reader&, const ModelDefinition&,
                                             const RankPlan&, Device);

} // namespace ninfer::models::qwen4_exp
