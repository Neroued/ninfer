#pragma once

#include "models/inkling2/execution/parameters.h"

namespace ninfer::models::inkling2::execution {

void routed_moe(const Parameters&, const LogicalLayerConfig&, const Tensor& input,
                ops::MoeTokenValidity, Tensor& output, MemoryArena&, CudaExecutionView);

} // namespace ninfer::models::inkling2::execution
