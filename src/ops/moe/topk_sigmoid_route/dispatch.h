#pragma once
#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail::moe_sigmoid {
void launch(const Tensor&, const Weight&, const Tensor&, ExpertRoutes&, const SigmoidRouteProfile&,
            float* scores, cudaStream_t);
}
