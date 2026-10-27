#pragma once

#include "ops/moe/kernels/cuda/groupwise_int/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/launch.h"

#include "ops/moe/kernels/cuda/block_scaled/clamped/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_routed {

namespace kernel = moe_cuda;

enum class Implementation { GroupwiseInt, BlockScaled, ClampedBlockScaled };

struct Plan {
    Implementation implementation = Implementation::GroupwiseInt;
    kernel::groupwise_int::Plan integer;
    kernel::block_scaled::Plan block_scaled;
    kernel::block_scaled::clamped::Plan clamped;
};

struct ShapePolicy {
    // A zero shared width denotes an absent shared term.
    Plan (*select)(const SwiGLUProfile&, int tokens, int shared_intermediate);
    std::span<const int> allocation_boundaries;
    int all_slice_tokens    = 4096;
    int masked_slice_tokens = 4096;

    int slice_tokens(ExpertSubsetKind subset) const {
        return subset == ExpertSubsetKind::MaskOnes || subset == ExpertSubsetKind::MaskZeros
                   ? masked_slice_tokens
                   : all_slice_tokens;
    }

    bool scaled_shared = true;
    bool accumulate    = true;
};

const ShapePolicy& shape_dispatch(const RoutedSwiGLUProfile& profile);
const ShapePolicy& q4_q5_h2048_i512_e256_k8();
const ShapePolicy& q4_q6_h2048_i512_e256_k8();
const ShapePolicy& q8_q8_h2048_i512_e256_k8();
const ShapePolicy& nvfp4_h2560_i640_e512_k10();
const ShapePolicy& fp8_h2560_i640_e512_k10();
const ShapePolicy& nvfp4_h4096_i2048_e288_k8();
const ShapePolicy& nvfp4_h4096_i1024_e288_k8();
const ShapePolicy& fp8_h4096_i2048_e288_k8();

} // namespace ninfer::ops::detail::moe_routed
