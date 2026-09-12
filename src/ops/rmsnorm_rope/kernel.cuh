#pragma once
#include "ops/common/dflash_rope.cuh"
#include "ops/rmsnorm_rope/d128.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops {
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
        dflash_rope_sincos(positions, token, pair, &sin_cache[pair], &cos_cache[pair]);
        weight_cache[pair] = weight[pair];
    }
    __syncthreads();
    const std::int64_t base = (static_cast<std::int64_t>(token) * (query ? 32 : 8) + head) * kPairs;
    const auto out    = detail::rmsnorm_rope_d128_head(data[base + lane], data[base + lane + 32],
                                                       weight_cache[lane], weight_cache[lane + 32],
                                                       cos_cache, sin_cache, lane);
    data[base + lane] = out.first;
    data[base + lane + 32] = out.second;
}
} // namespace ninfer::ops
