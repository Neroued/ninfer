#pragma once
#include "ninfer/ops/moe.h"
#include "ops/moe/kernels/cuda/launch.h"

namespace ninfer::ops::detail::moe_sigmoid {
void launch(const Tensor&, const Weight&, const Tensor&, ExpertRoutes&, const SigmoidRouteProfile&,
            float* scores, moe_cuda::TokenValidity, cudaStream_t);
}
