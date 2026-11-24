#pragma once

#include "ops/moe/kernels/cuda/block_scaled/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8::block64 {
void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& storage, CudaExecutionView execution);
void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, bool shared, MoeEpilogue mode, Tensor& destination,
                   const Storage& storage, CudaExecutionView execution);
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::fp8::block64
