#include "ninfer/ops/moe.h"

#include <stdexcept>

namespace ninfer::ops {

std::size_t routed_swiglu_workspace_capacity_bytes(const RoutedSwiGLUProfile&, ExpertSubsetKind,
                                                   std::int32_t, std::int32_t) {
    throw std::invalid_argument("routed_swiglu CPU: no implementation for dgx_spark");
}

void routed_swiglu(const Tensor&, const RoutedSwiGLUProfile&, const CpuSwiGLUExperts&,
                   const ExpertRoutes&, ExpertSubset, Tensor&, MemoryArena&, CpuExecutionView) {
    throw std::invalid_argument("routed_swiglu CPU: no implementation for dgx_spark");
}

} // namespace ninfer::ops
