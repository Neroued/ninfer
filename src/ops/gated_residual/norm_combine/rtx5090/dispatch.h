#pragma once
#include "ops/gated_residual/operands.h"

namespace ninfer::ops::detail::hc {
void launch_h2560(Form, Operands, CudaExecutionView);
void launch_h5120(Form, Operands, CudaExecutionView);
} // namespace ninfer::ops::detail::hc
