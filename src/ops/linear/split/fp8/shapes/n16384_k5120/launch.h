#pragma once
#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/fp8/fp8_a8_plan.h"

namespace ninfer::ops::detail::linear_split::fp8_n16384_k5120 {
void launch_a16(const Tensor&, const Weight&, std::span<const Tensor>, cudaStream_t);
void launch_a8(const Tensor&, const Weight&, std::span<const Tensor>, Fp8A8Workspace, cudaStream_t);
std::size_t partial_capacity(std::int32_t max_columns);
} // namespace ninfer::ops::detail::linear_split::fp8_n16384_k5120
