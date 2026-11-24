#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <span>

namespace ninfer::ops {

// destination=BF16(sum(contributions) + optional residual), with FP32 addition and one final
// cast. Accepts one or two contiguous FP32 tensors, a BF16 destination, and optional BF16 residual
// of the same positive shape, all on execution.device(). Residual may equal destination exactly;
// other overlap with destination is forbidden. Without residual the old destination is not read. No
// scratch.
void sum_contributions(std::span<const Tensor> contributions, const Tensor* residual,
                       Tensor& destination, CudaExecutionView execution);

} // namespace ninfer::ops
