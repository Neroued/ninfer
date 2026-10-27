#pragma once

#include "ops/common/cuda/reduce.cuh"
#include "ops/common/quant/fp8.cuh"

namespace ninfer::ops::detail::kpool {

// A full warp owns features 4*lane + {0,1,2,3}. The basis belongs to the
// Indexer; G128/UE8M0 encoding remains the common cache codec.
__device__ __forceinline__ void index_hadamard128(float (&values)[4]) {
#pragma unroll
    for (int stride = 1; stride < 4; stride *= 2) {
#pragma unroll
        for (int start = 0; start < 4; start += 2 * stride) {
#pragma unroll
            for (int j = 0; j < stride; ++j) {
                const float a = values[start + j], b = values[start + j + stride];
                values[start + j]          = a + b;
                values[start + j + stride] = a - b;
            }
        }
    }
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int stride = 1; stride < 32; stride *= 2) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = __shfl_xor_sync(0xffffffffu, values[j], stride);
            values[j]         = lane & stride ? other - values[j] : values[j] + other;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) values[j] *= 0.08838834764831845F;
}

struct EncodedIndexLane {
    unsigned codes;
    std::uint8_t scale;
};

template <bool Query>
__device__ __forceinline__ EncodedIndexLane encode_index_lane(float (&values)[4]) {
    index_hadamard128(values);
    float maximum = 0.F;
#pragma unroll
    for (int j = 0; j < 4; ++j) maximum = fmaxf(maximum, fabsf(values[j]));
    maximum = common::warp_max(maximum);
    // Only the private query activation uses the reference's amax floor.
    if constexpr (Query) maximum = fmaxf(maximum, 1e-4F);
    const auto scale  = common::fp8_ue8m0_scale(maximum);
    const unsigned lo = common::fp8_quant_code2(values[0], values[1], scale.inverse_scale);
    const unsigned hi = common::fp8_quant_code2(values[2], values[3], scale.inverse_scale);
    return {lo | (hi << 16), scale.scale};
}
} // namespace ninfer::ops::detail::kpool
