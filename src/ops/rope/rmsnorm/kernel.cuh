#pragma once

#include "ops/common/cuda/reduce.cuh"
#include "ops/rope/kernels/phase.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {

// A CTA shares phases across Q and K heads of one token. Each (half) warp owns
// a complete head and retains the unrounded normalized values in registers.
template <int QueryHeads, int KeyHeads, int HeadsPerBlock, int LanesPerHead, bool WideAngles>
__global__ __launch_bounds__(HeadsPerBlock* LanesPerHead) void rmsnorm_rope_d128_kernel(
    const std::int32_t* __restrict__ positions, const __nv_bfloat162* __restrict__ q_weight,
    const __nv_bfloat162* __restrict__ k_weight, __nv_bfloat162* __restrict__ q,
    __nv_bfloat162* __restrict__ k, float epsilon, RopeD128Frequencies frequencies) {
    constexpr int Groups        = (QueryHeads + KeyHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int Items         = 32 / LanesPerHead;
    constexpr bool CacheWeights = QueryHeads == 32 && HeadsPerBlock == 8 && LanesPerHead == 32;
    static_assert(LanesPerHead == 16 || LanesPerHead == 32);
    static_assert(HeadsPerBlock * LanesPerHead >= 64);
    const int token      = blockIdx.x / Groups;
    const int head       = (blockIdx.x % Groups) * HeadsPerBlock + threadIdx.x / LanesPerHead;
    const int lane       = threadIdx.x % LanesPerHead;
    const bool query     = head < QueryHeads;
    const int local_head = query ? head : head - QueryHeads;
    auto* data           = query ? q : k;
    const auto* weight   = query ? q_weight : k_weight;
    const int base       = (token * (query ? QueryHeads : KeyHeads) + local_head) * 64;
    __shared__ float sine[64], cosine[64];
    __shared__ __nv_bfloat162 weight_cache[CacheWeights ? 64 : 1];
    if constexpr (CacheWeights)
        if (threadIdx.x < 64) weight_cache[threadIdx.x] = weight[threadIdx.x];
    if (threadIdx.x < 64) {
        const int pair     = threadIdx.x;
        const double angle = double(positions[token]) * frequencies.value[pair];
        // theta>=1 bounds phases by the I32 position range. Keep the large-angle
        // library's register cost out of these common profiles.
        if constexpr (!WideAngles) {
            const float reduced =
                float(angle - nearbyint(angle * 0.159154943091895336) * 6.28318530717958648);
            sincosf(reduced, sine + pair, cosine + pair);
        } else {
            double s, c;
            sincos(angle, &s, &c);
            sine[pair]   = float(s);
            cosine[pair] = float(c);
        }
    }
    if constexpr (CacheWeights) __syncthreads();
    float2 first[Items]{}, second[Items]{}, gain0[Items]{}, gain1[Items]{};
    float square = 0;
    if (head < QueryHeads + KeyHeads) {
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const int i = lane + j * LanesPerHead;
            first[j]    = __bfloat1622float2(data[base + i]);
            second[j]   = __bfloat1622float2(data[base + i + 32]);
            gain0[j]    = __bfloat1622float2(CacheWeights ? weight_cache[i] : weight[i]);
            gain1[j]    = __bfloat1622float2(CacheWeights ? weight_cache[i + 32] : weight[i + 32]);
            square += first[j].x * first[j].x + first[j].y * first[j].y +
                      second[j].x * second[j].x + second[j].y * second[j].y;
        }
    }
    square        = common::warp_reduce_sum<LanesPerHead>(square);
    float inverse = lane == 0 ? rsqrtf(square * (1.F / 128.F) + epsilon) : 0.F;
    inverse       = __shfl_sync(common::kFullWarpMask, inverse, 0, LanesPerHead);
    if constexpr (!CacheWeights) __syncthreads();
    if (head < QueryHeads + KeyHeads) {
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const int i         = lane + j * LanesPerHead;
            const int pair      = i * 2;
            const float x       = first[j].x * inverse * gain0[j].x;
            const float y       = first[j].y * inverse * gain0[j].y;
            const float u       = second[j].x * inverse * gain1[j].x;
            const float v       = second[j].y * inverse * gain1[j].y;
            data[base + i]      = __floats2bfloat162_rn(x * cosine[pair] - u * sine[pair],
                                                        y * cosine[pair + 1] - v * sine[pair + 1]);
            data[base + i + 32] = __floats2bfloat162_rn(u * cosine[pair] + x * sine[pair],
                                                        v * cosine[pair + 1] + y * sine[pair + 1]);
        }
    }
}
} // namespace ninfer::ops::detail
