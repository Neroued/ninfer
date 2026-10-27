#pragma once

#include "ops/gated_residual/mix/bf16/workspace.h"

namespace ninfer::ops::detail::hc::mix::bf16 {
void gemv_down(Kind kind, Operands operands, Scratch scratch, cudaStream_t stream);
void gemv_up(Operands operands, Scratch scratch, cudaStream_t stream);
} // namespace ninfer::ops::detail::hc::mix::bf16
