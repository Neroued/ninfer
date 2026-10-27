#pragma once

#include "ops/gated_residual/mix/bf16/workspace.h"
#include "ops/gated_residual/mix/bf16/select.h"

namespace ninfer::ops::detail::hc::mix::bf16 {
void launch(Kind kind, Operands operands, Plan plan, Scratch scratch, CudaExecutionView execution);
} // namespace ninfer::ops::detail::hc::mix::bf16
