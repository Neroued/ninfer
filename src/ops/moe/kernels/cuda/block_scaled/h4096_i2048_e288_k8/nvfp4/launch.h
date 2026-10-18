#pragma once

#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::nvfp4 {

void launch_input(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                  ExpertSubset subset, const Storage& storage, CudaExecutionView execution,
                  bool clear_counts);
void launch_experts(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                    const Plan& plan, const Storage& storage, CudaExecutionView execution);
void launch_vector(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                   ExpertSubset subset, MoeEpilogue mode, Tensor& destination,
                   const Storage& storage, const Plan& plan, CudaExecutionView execution);
} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::nvfp4
