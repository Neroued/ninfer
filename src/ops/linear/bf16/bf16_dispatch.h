#pragma once

#include "core/weight.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t);
Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy);

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream);

} // namespace ninfer::ops::detail
