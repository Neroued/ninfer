#pragma once

#include "ops/gated_residual/mix/bf16/operands.h"

namespace ninfer::ops::detail::hc::mix::bf16 {

void execute(Weights weights, const Tensor& normalized, MemoryArena& workspace, Tensor& output,
             Tensor* injection, CudaExecutionView execution);
std::size_t capacity(Weights weights, std::int32_t first, std::int32_t last);

} // namespace ninfer::ops::detail::hc::mix::bf16
