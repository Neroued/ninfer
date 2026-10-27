#pragma once

#include "ops/moe/kernels/cuda/block_scaled/clamped/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_sigmoid {
struct ShapePolicy {
    moe_cuda::block_scaled::clamped::Plan (*select)(const SwiGLUProfile&, int tokens);
    std::span<const int> allocation_boundaries;
    int slice_tokens;
};

const ShapePolicy& nvfp4_h4096_i1024_e288_k8();
} // namespace ninfer::ops::detail::moe_sigmoid
