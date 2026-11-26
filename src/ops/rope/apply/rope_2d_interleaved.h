#pragma once

#include "core/device.h"

namespace ninfer {
struct Tensor;

namespace ops::detail {

void launch_rope_2d_interleaved(const Tensor& positions, float theta, Tensor& q, Tensor* k,
                                CudaExecutionView execution);

} // namespace ops::detail
} // namespace ninfer
