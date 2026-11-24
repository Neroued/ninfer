#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops {

/**
 * destination[i] = round_to_bf16_rne(source[i]).
 *
 * Source is FP32 and destination is BF16, with identical shapes and contiguous, non-null
 * storage. They do not overlap. Converts every element independently with round-to-nearest-even,
 * writes all of destination and preserves source. No workspace or other state is used.
 */
void cast_fp32_to_bf16(const Tensor& source, Tensor& destination, CudaExecutionView execution);

} // namespace ninfer::ops
