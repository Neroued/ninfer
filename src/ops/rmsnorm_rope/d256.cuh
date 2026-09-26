#pragma once

#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

// One warp owns one represented BF16 D256 head. Lane l carries the pairs l, l+32, l+64, l+96, the
// layout rmsnorm_warp_bf16x2_kernel uses, so the sum of squares accumulates in the same order and
// the epilogue is the same helper: the normalized value is bit-identical to the standalone norm.
struct RmsnormRopeD256Head {
    __nv_bfloat162 pair[4];
};

__device__ __forceinline__ RmsnormRopeD256Head rmsnorm_rope_d256_normalize(
    const __nv_bfloat162* __restrict__ input, const __nv_bfloat162* __restrict__ weight,
    std::int64_t base, int lane) {
    constexpr int kHeadDim   = 256;
    constexpr float kEpsilon = 1.0e-6F;
    __nv_bfloat162 values[4];
    __nv_bfloat162 weights[4];
    float sum = 0.0F;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int pair  = lane + k * 32;
        values[k]       = input[base + pair];
        weights[k]      = weight[pair];
        const float2 xf = __bfloat1622float2(values[k]);
        sum += xf.x * xf.x + xf.y * xf.y;
    }
    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(kHeadDim) + kEpsilon) : 0.0F;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

    RmsnormRopeD256Head out;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 xf = __bfloat1622float2(values[k]);
        const float2 wf = __bfloat1622float2(weights[k]);
        out.pair[k] =
            __floats2bfloat162_rn(rmsnorm_epilogue<RmsEpilogue::Offset>(xf.x, inv, wf.x, 0.0F),
                                  rmsnorm_epilogue<RmsEpilogue::Offset>(xf.y, inv, wf.y, 0.0F));
    }
    return out;
}

// Split-half rotation over the first 64 channels, which is what R=64 means for a 256-wide head:
// channel p pairs with p + 32. The norm layout keeps those two in different lanes, so the partner
// arrives through __shfl_xor_sync(..., 16) and the coefficients are indexed by lane & 15 - exactly
// the ones lane p < 16 receives in the standalone rope kernel.
__device__ __forceinline__ __nv_bfloat162 rmsnorm_rope_d256_rotate(__nv_bfloat162 normalized,
                                                                   float c0, float c1, float s0,
                                                                   float s1, int lane) {
    constexpr int kHalfPair     = 16;
    const __nv_bfloat162 theirs = __shfl_xor_sync(kFullWarpMask, normalized, kHalfPair);
    const float2 first =
        lane < kHalfPair ? __bfloat1622float2(normalized) : __bfloat1622float2(theirs);
    const float2 second =
        lane < kHalfPair ? __bfloat1622float2(theirs) : __bfloat1622float2(normalized);
    if (lane < kHalfPair) {
        return __floats2bfloat162_rn(first.x * c0 - second.x * s0, first.y * c1 - second.y * s1);
    }
    return __floats2bfloat162_rn(second.x * c0 + first.x * s0, second.y * c1 + first.y * s1);
}

} // namespace ninfer::ops::detail
