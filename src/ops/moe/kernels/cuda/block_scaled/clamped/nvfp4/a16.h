#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4 {

// Grouping and output initialization precede this call on the same stream.
// The caller owns all intermediate and output storage.
void launch_a16_experts(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, const Plan&,
                        const Storage&, CudaExecutionView);

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
