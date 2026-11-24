#pragma once

#include "ops/gated_residual/mix/workspace.h"
#include "ops/gated_residual/mix/select.h"

namespace ninfer::ops::detail::hc::mix {
void launch(Kind kind, Operands operands, Plan plan, Scratch scratch, CudaExecutionView execution);
} // namespace ninfer::ops::detail::hc::mix
