#pragma once
#include "ops/rope/kernels/phase.cuh"
#include "ops/common/cuda/reduce.cuh"
#include <cuda_bf16.h>
#include <cstdint>

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
    sum           = common::warp_reduce_sum(sum);
    float inverse = lane == 0 ? rsqrtf(sum * (1.0F / 128.0F) + kEpsilon) : 0.0F;
    inverse       = __shfl_sync(common::kFullWarpMask, inverse, 0);

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

// A CTA owns eight heads of one token. Pair form uses four Q CTAs and one K CTA;
// the single-K form uses one CTA. Each warp evaluates one complete head.
template <bool Pair>
__global__ __launch_bounds__(256) void rmsnorm_rope_d128_kernel(
    const std::int32_t* __restrict__ positions, const __nv_bfloat16* __restrict__ q_norm,
    const __nv_bfloat16* __restrict__ k_norm, __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k) {
    constexpr int kPairs = 64;
    const int token      = blockIdx.x;
    const bool query     = Pair && blockIdx.y < 4;
    const int head =
        (query ? static_cast<int>(blockIdx.y) * 8 : 0) + static_cast<int>(threadIdx.x) / 32;
    const int lane     = threadIdx.x % 32;
    auto* data         = reinterpret_cast<__nv_bfloat162*>(query ? q : k);
    const auto* weight = reinterpret_cast<const __nv_bfloat162*>(query ? q_norm : k_norm);
    __shared__ float cos_cache[kPairs];
    __shared__ float sin_cache[kPairs];
    __shared__ __nv_bfloat162 weight_cache[kPairs];
    if (threadIdx.x < kPairs) {
        const int pair = threadIdx.x;
        rope_d128_theta1e7_sincos(positions, token, pair, &sin_cache[pair], &cos_cache[pair]);
        weight_cache[pair] = weight[pair];
    }
    __syncthreads();
    const std::int64_t base = (static_cast<std::int64_t>(token) * (query ? 32 : 8) + head) * kPairs;
    const auto out =
        rmsnorm_rope_d128_head(data[base + lane], data[base + lane + 32], weight_cache[lane],
                               weight_cache[lane + 32], cos_cache, sin_cache, lane);
    data[base + lane]      = out.first;
    data[base + lane + 32] = out.second;
}
} // namespace ninfer::ops::detail
