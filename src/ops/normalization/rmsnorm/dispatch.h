#pragma once

#include "ops/normalization/rmsnorm/operands.h"
#include "core/device.h"

namespace ninfer::ops::detail::rmsnorm {

void dispatch(const Tensor& x, const Tensor& weight, const Tensor* gate, float eps, Epilogue kind,
              Tensor& out, Operands operands, CudaExecutionView execution);

} // namespace ninfer::ops::detail::rmsnorm
