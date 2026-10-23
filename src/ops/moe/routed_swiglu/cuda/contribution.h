#pragma once

#include "ninfer/ops/moe.h"

namespace ninfer::ops::detail {

// Private composition view. A zero validity entry denotes a mathematical zero;
// its values column is not readable. A null validity plane denotes dense values.
// Both planes are borrowed until the consumer finishes on the declared stream.
struct RoutedSwiGLUContribution {
    const float* values;
    const int* valid_tokens;
};

namespace moe_routed {
std::size_t contribution_workspace_capacity_bytes(const RoutedSwiGLUProfile&, ExpertSubsetKind,
                                                  int first, int last);

// The complete MoE owns FP32[H,T] values and I32[T] validity outside this call's
// workspace. They must be disjoint from the current workspace and read operands.
// Registered clamped native profiles and non-None subsets only. Writes validity
// on every replay; values are written wherever the returned view permits reads.
RoutedSwiGLUContribution execute_contribution(const Tensor&, const RoutedSwiGLUProfile&,
                                              const CudaSwiGLUExperts&, const ExpertRoutes&,
                                              ExpertSubset, Tensor& values, int* valid_tokens,
                                              MemoryArena&, CudaExecutionView);
} // namespace moe_routed
} // namespace ninfer::ops::detail
