#pragma once

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

struct RmsnormRopeD128Pair {
    __nv_bfloat162 first;
    __nv_bfloat162 second;
};

// One warp owns one represented BF16 D128 head. Each lane carries adjacent pairs from the first
// and second split halves and returns the corresponding full-head RoPE pairs.
__device__ __forceinline__ RmsnormRopeD128Pair rmsnorm_rope_d128_head(
    __nv_bfloat162 input0, __nv_bfloat162 input1, __nv_bfloat162 weight0, __nv_bfloat162 weight1,
    const float* cos_cache, const float* sin_cache, int lane) {
    constexpr float kEpsilon = 1.0e-6F;
    const float2 input0_f32  = __bfloat1622float2(input0);
    const float2 input1_f32  = __bfloat1622float2(input1);
    float sum                = input0_f32.x * input0_f32.x + input0_f32.y * input0_f32.y +
                input1_f32.x * input1_f32.x + input1_f32.y * input1_f32.y;
    sum           = warp_reduce_sum(sum);
    float inverse = lane == 0 ? rsqrtf(sum * (1.0F / 128.0F) + kEpsilon) : 0.0F;
    inverse       = __shfl_sync(kFullWarpMask, inverse, 0);

    const float2 weight0_f32  = __bfloat1622float2(weight0);
    const float2 weight1_f32  = __bfloat1622float2(weight1);
    const float normalized0_x = input0_f32.x * inverse * weight0_f32.x;
    const float normalized0_y = input0_f32.y * inverse * weight0_f32.y;
    const float normalized1_x = input1_f32.x * inverse * weight1_f32.x;
    const float normalized1_y = input1_f32.y * inverse * weight1_f32.y;
    const int rotary_pair     = lane * 2;
    const float cosine0       = cos_cache[rotary_pair];
    const float cosine1       = cos_cache[rotary_pair + 1];
    const float sine0         = sin_cache[rotary_pair];
    const float sine1         = sin_cache[rotary_pair + 1];
    return {
        __floats2bfloat162_rn(normalized0_x * cosine0 - normalized1_x * sine0,
                              normalized0_y * cosine1 - normalized1_y * sine1),
        __floats2bfloat162_rn(normalized1_x * cosine0 + normalized0_x * sine0,
                              normalized1_y * cosine1 + normalized0_y * sine1),
    };
}

} // namespace ninfer::ops::detail
