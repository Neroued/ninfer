#pragma once

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kRmsnormPackTailRows        = 5120;
inline constexpr int kRmsnormPackTailPairsPerRow = kRmsnormPackTailRows / 2;
inline constexpr float kRmsnormPackTailEps       = 1.0e-6F;

template <int kBlock>
__device__ __forceinline__ float rmsnorm_pack_tail_inverse(float local_sum, float* warp_sums,
                                                           float* inverse) {
    constexpr int kWarps = kBlock / kWarpSize;
    const int lane       = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp       = static_cast<int>(threadIdx.x) / kWarpSize;

    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) { warp_sums[warp] = local_sum; }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < kWarps ? warp_sums[lane] : 0.0F;
        block_sum       = warp_reduce_sum(block_sum);
        if (lane == 0) {
            *inverse = rsqrtf(block_sum * (1.0F / kRmsnormPackTailRows) + kRmsnormPackTailEps);
        }
    }
    __syncthreads();
    return *inverse;
}

// Grid x selects a mask column, grid y its request. Anchor columns are never read.
template <int kBlock>
__global__ __launch_bounds__(kBlock) void rmsnorm_pack_tail_kernel(
    const __nv_bfloat162* __restrict__ input, const __nv_bfloat162* __restrict__ weight,
    __nv_bfloat162* __restrict__ output, int width) {
    constexpr int kPairsPerThread = kRmsnormPackTailPairsPerRow / kBlock;
    static_assert(kRmsnormPackTailPairsPerRow % kBlock == 0);

    const int tail  = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    const std::int64_t input_base =
        static_cast<std::int64_t>(batch * width + tail + 1) * kRmsnormPackTailPairsPerRow;
    const std::int64_t output_base =
        static_cast<std::int64_t>(batch * (width - 1) + tail) * kRmsnormPackTailPairsPerRow;

    __nv_bfloat162 values[kPairsPerThread];
    __nv_bfloat162 weights[kPairsPerThread];
    float local_sum = 0.0F;
#pragma unroll
    for (int item = 0; item < kPairsPerThread; ++item) {
        const int pair = static_cast<int>(threadIdx.x) + item * kBlock;
        values[item]   = input[input_base + pair];
        weights[item]  = weight[pair];
        const float2 x = __bfloat1622float2(values[item]);
        local_sum += x.x * x.x + x.y * x.y;
    }

    __shared__ float warp_sums[kBlock / kWarpSize];
    __shared__ float inverse;
    const float inv = rmsnorm_pack_tail_inverse<kBlock>(local_sum, warp_sums, &inverse);

#pragma unroll
    for (int item = 0; item < kPairsPerThread; ++item) {
        const int pair             = static_cast<int>(threadIdx.x) + item * kBlock;
        const float2 x             = __bfloat1622float2(values[item]);
        const float2 w             = __bfloat1622float2(weights[item]);
        output[output_base + pair] = __floats2bfloat162_rn(x.x * inv * w.x, x.y * inv * w.y);
    }
}

} // namespace ninfer::ops
