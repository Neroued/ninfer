#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <span>

namespace ninfer::ops {

// destination=BF16(sum(contributions) + optional residual), with FP32 addition and one final
// cast. Accepts one or two contiguous FP32 tensors, a BF16 destination, and optional BF16 residual
// of the same positive shape, all on execution.device(). Residual may equal destination exactly;
// other overlap with destination is forbidden. Without residual the old destination is not read. No
// scratch. Empty valid_columns sums every token. Otherwise it is Device I32[P], tokens are
// the flattened trailing dimensions after ne[0], T=P*tokens_per_row and each row has a valid
// prefix in [0,tokens_per_row]. Inactive tokens read no contribution and copy residual bits
// exactly (or write zero without residual). The validity plane is disjoint from destination.
void sum_contributions(std::span<const Tensor> contributions, const Tensor* residual,
                       Tensor& destination, const Tensor& valid_columns, int tokens_per_row,
                       CudaExecutionView execution);

} // namespace ninfer::ops
