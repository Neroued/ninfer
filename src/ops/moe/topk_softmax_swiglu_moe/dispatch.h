#pragma once

#include "ops/moe/kernels/cuda/groupwise_int/launch.h"
#include "ops/moe/kernels/cuda/routing/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/launch.h"

#include <span>

namespace ninfer::ops::detail::moe_complete {

namespace kernel = moe_cuda;
using kernel::routing::RouterAlgorithm;

struct Plan {
    kernel::ExpertImplementation implementation = kernel::ExpertImplementation::GroupwiseInt;
    kernel::routing::RouterPlan router;
    kernel::groupwise_int::Plan integer;
    kernel::block_scaled::Plan block_scaled;
};

struct ShapePolicy {
    Plan (*select)(int tokens, int shared_intermediate, ExpertSubsetKind subset);
    std::span<const int> allocation_boundaries;
    int masked_slice_tokens = 4096;

    int slice_tokens(ExpertSubsetKind subset) const {
        return subset == ExpertSubsetKind::MaskOnes || subset == ExpertSubsetKind::MaskZeros
                   ? masked_slice_tokens
                   : 4096;
    }
};

const ShapePolicy& shape_dispatch(const MoeProfile& profile);
const ShapePolicy& q4_q5_h2048_i512_e256_k8();
const ShapePolicy& q4_q6_h2048_i512_e256_k8();
const ShapePolicy& q8_q8_h2048_i512_e256_k8();
const ShapePolicy& nvfp4_h2560_i640_e512_k10();
const ShapePolicy& nvfp4_h2560_i320_e512_k10();
const ShapePolicy& fp8_h2560_i640_e512_k10();
const ShapePolicy& fp8_h2560_i320_e512_k10();

} // namespace ninfer::ops::detail::moe_complete
