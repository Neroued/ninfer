#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4 {

void launch_mul_input(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, ExpertSubset,
                      const Plan&, const Storage&, CudaExecutionView);
// After small grouping, quantizes already indexed rows. After counted grouping,
// publishes the assignment indices and quantizes those rows in the same launch.
void launch_mul_grouped_input(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&,
                              ExpertSubset, const Plan&, const Storage&, CudaExecutionView);
void launch_mul_experts(const Tensor&, const CudaSwiGLUExperts&, const ExpertRoutes&, const Plan&,
                        const Storage&, CudaExecutionView);

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::clamped::nvfp4
