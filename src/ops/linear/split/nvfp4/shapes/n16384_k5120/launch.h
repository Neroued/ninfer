#pragma once
#include "ops/linear/split/dispatch.h"
#include "ops/linear/kernels/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear/kernels/nvfp4/nvfp4_operands.h"

namespace ninfer::ops::detail::linear_split::nvfp4_n16384_k5120 {
void launch_a16(const Tensor&, const Weight&, std::span<const Tensor>, cudaStream_t);
void launch_a4(const Tensor&, const Weight&, std::span<const Tensor>, Nvfp4A4Workspace,
               cudaStream_t);
void launch_a4_tma(const Nvfp4A4Operands&, std::span<const Tensor>, cudaStream_t);
} // namespace ninfer::ops::detail::linear_split::nvfp4_n16384_k5120
