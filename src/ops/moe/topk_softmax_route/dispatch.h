#pragma once

#include "ops/moe/kernels/cuda/routing/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_route {

inline constexpr int kSliceTokens = 4096;
using moe_cuda::routing::RouterPlan;
using moe_cuda::routing::RouterAlgorithm;
RouterPlan select_h2048_e256_k8(int tokens);
std::span<const int> allocation_boundaries();
RouterPlan select_h2560_e512_k10(int tokens);
} // namespace ninfer::ops::detail::moe_route
