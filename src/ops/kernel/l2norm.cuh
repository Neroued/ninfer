#pragma once

// ninfer::ops - L2Norm kernels over contiguous BF16 rows.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// Fast domain: D in {64, 128, 192, 256}. One warp owns one row and keeps the input in registers.
template <int Block>
__launch_bounds__(Block) __global__
    void l2norm_warp_bf16x2_kernel(const __nv_bfloat162* x, __nv_bfloat162* out, std::int32_t d,
                                   std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock   = Block / kWarpSize;
    constexpr int kMaxPairsPerLane = 4;
    const int lane                 = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp                 = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row         = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            values[k]       = x[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            const float2 xf      = __bfloat1622float2(values[k]);
            out[row_base + pair] = __floats2bfloat162_rn(xf.x * inv, xf.y * inv);
        }
    }
}

// Functional fallback for unaligned data and widths outside the aligned fast domain.
__launch_bounds__(512) __global__
    void l2norm_generic_kernel(const __nv_bfloat16* x, __nv_bfloat16* out, std::int32_t d,
                               std::int64_t rows, float eps) {
    constexpr int kRowsPerBlock = 512 / kWarpSize;
    const int lane              = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp              = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row      = static_cast<std::int64_t>(blockIdx.x) * kRowsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    float sum               = 0.0f;
    for (std::int64_t i = lane; i < static_cast<std::int64_t>(d); i += kWarpSize) {
        const float value = __bfloat162float(x[base + i]);
        sum += value * value;
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);
    for (std::int64_t i = lane; i < static_cast<std::int64_t>(d); i += kWarpSize) {
        const std::int64_t index = base + i;
        out[index]               = __float2bfloat16_rn(__bfloat162float(x[index]) * inv);
    }
}

} // namespace ninfer::ops
