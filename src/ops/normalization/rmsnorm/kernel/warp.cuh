#pragma once

#include "ops/common/cuda/reduce.cuh"
#include "ops/normalization/rmsnorm/kernel/epilogue.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::rmsnorm {
// Fast geometry for D in {64, 128, 192, 256}. One warp owns one row, keeps the input in
// registers, and uses only warp shuffles for the reduction. Block is a scheduling choice rather
// than part of the row geometry.
template <Epilogue Kind, int Block, bool Prefetch, int FixedD = 0>
__launch_bounds__(Block) __global__
    void rmsnorm_warp_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out,
                                    std::int32_t input_d, std::int64_t rows, float eps) {
    const int d = FixedD ? FixedD : input_d;
    static_assert(Block % common::kWarpSize == 0);
    constexpr int kWarpsPerBlock   = Block / common::kWarpSize;
    constexpr int kMaxPairsPerLane = 4;
    const int lane                 = static_cast<int>(threadIdx.x) & (common::kWarpSize - 1);
    const int warp                 = static_cast<int>(threadIdx.x) / common::kWarpSize;
    const std::int64_t row         = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[kMaxPairsPerLane];
    __nv_bfloat162 weights[kMaxPairsPerLane];
    __nv_bfloat162 gates[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * common::kWarpSize;
        if (pair < pairs) {
            values[k] = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[k] = weight[pair];
                if constexpr (is_gated<Kind>) { gates[k] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = common::warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(d) + eps) : 0.0f;
    inv       = __shfl_sync(common::kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * common::kWarpSize;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            __nv_bfloat162 w_pair;
            __nv_bfloat162 z_pair{};
            if constexpr (Prefetch) {
                w_pair = weights[k];
                if constexpr (is_gated<Kind>) { z_pair = gates[k]; }
            } else {
                w_pair = weight[pair];
                if constexpr (is_gated<Kind>) { z_pair = z[row_base + pair]; }
            }
            const float2 wf = __bfloat1622float2(w_pair);
            float2 zf{0.0f, 0.0f};
            if constexpr (is_gated<Kind>) { zf = __bfloat1622float2(z_pair); }
            out[row_base + pair] =
                __floats2bfloat162_rn(apply_epilogue<Kind>(xf.x, inv, wf.x, zf.x),
                                      apply_epilogue<Kind>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Implements: include/ninfer/ops/rmsnorm.h
// Match: aligned contiguous BF16, plain or gated epilogue, D=128, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per lane; one warp owns one logical row.
template <Epilogue Kind, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_d128_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int64_t rows,
                                    float eps) {
    static_assert(Block % common::kWarpSize == 0);
    constexpr int kWarpsPerBlock = Block / common::kWarpSize;
    constexpr int kPairsPerRow   = 64;
    const int lane               = static_cast<int>(threadIdx.x) & (common::kWarpSize - 1);
    const int warp               = static_cast<int>(threadIdx.x) / common::kWarpSize;
    const std::int64_t row       = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = lane;
    const int pair1             = lane + common::kWarpSize;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    float sum                   = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;
    sum                         = common::warp_reduce_sum(sum);
    float inv                   = lane == 0 ? rsqrtf(sum * (1.0f / 128.0f) + eps) : 0.0f;
    inv                         = __shfl_sync(common::kFullWarpMask, inv, 0);

    const float2 w0 = __bfloat1622float2(weight[pair0]);
    const float2 w1 = __bfloat1622float2(weight[pair1]);
    float2 z0{0.0f, 0.0f};
    float2 z1{0.0f, 0.0f};
    if constexpr (is_gated<Kind>) {
        z0 = __bfloat1622float2(z[row_base + pair0]);
        z1 = __bfloat1622float2(z[row_base + pair1]);
    }
    out[row_base + pair0] = __floats2bfloat162_rn(apply_epilogue<Kind>(x0.x, inv, w0.x, z0.x),
                                                  apply_epilogue<Kind>(x0.y, inv, w0.y, z0.y));
    out[row_base + pair1] = __floats2bfloat162_rn(apply_epilogue<Kind>(x1.x, inv, w1.x, z1.x),
                                                  apply_epilogue<Kind>(x1.y, inv, w1.y, z1.y));
}

} // namespace ninfer::ops::detail::rmsnorm
