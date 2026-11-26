#pragma once
#include "ninfer/ops/moe.h"
#include "ops/moe/kernels/cuda/launch.h"
#include <cstddef>

namespace ninfer::ops::detail::moe_sigmoid {
extern const RouteProfile registered_route;
std::size_t workspace_capacity_bytes(int first, int last);
void launch(const Tensor&, const Weight&, const Tensor&, ExpertRoutes&, const SigmoidRouteProfile&,
            float* scores, moe_cuda::TokenValidity, cudaStream_t);
} // namespace ninfer::ops::detail::moe_sigmoid
