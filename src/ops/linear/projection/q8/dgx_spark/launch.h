#pragma once

#include "ops/linear/kernels/q8/q8_launch.h"

namespace ninfer::ops::detail {

void launch_q8_n2560_k6144_prefill512(const Tensor& x, const Weight& weight, Tensor& out,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
