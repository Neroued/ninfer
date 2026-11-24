#pragma once

#include "ops/common/cuda/reduce.cuh"
#include "ops/normalization/rmsnorm/kernel/epilogue.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::rmsnorm {

// One CTA owns a wide row. Vector routes require a width divisible by the CTA pair span.
template <Epilogue Kind, int Block, int MaxPairsPerThread, bool Prefetch, int FixedD = 0>
__launch_bounds__(Block) __global__
    void rmsnorm_cta_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                   const __nv_bfloat162* z, __nv_bfloat162* out,
                                   std::int32_t input_d, std::int64_t rows, float eps) {
    const int d = FixedD ? FixedD : input_d;
    static_assert(Block % common::kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    // Neither the weight nor the gate depends on the sum of squares, so under Prefetch both
    // are issued in the same pass as x and the two trips to memory overlap instead of running
    // back to back across the reduction. They cost registers, so the launcher turns this off
    // once the grid is wide enough that occupancy is worth more than the overlap.
    __nv_bfloat162 weights[MaxPairsPerThread];
    __nv_bfloat162 gates[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair = static_cast<int>(threadIdx.x) + k * Block;
            values[k]      = x[row_base + pair];
            if constexpr (Prefetch) {
                weights[k] = weight[pair];
                if constexpr (is_gated<Kind>) { gates[k] = z[row_base + pair]; }
            }
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / common::kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = common::block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
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
// Match: aligned contiguous BF16, plain epilogue, D=2048, sm_120a.
// Algorithm assumptions: exactly two BF16x2 values per thread; one 512-thread CTA owns one row.
template <Epilogue Kind>
__launch_bounds__(512) __global__
    void rmsnorm_d2048_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                     const __nv_bfloat162* z, __nv_bfloat162* out,
                                     std::int64_t rows, float eps) {
    constexpr int kBlock       = 512;
    constexpr int kPairsPerRow = 1024;
    const std::int64_t row     = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t row_base = row * kPairsPerRow;
    const int pair0             = static_cast<int>(threadIdx.x);
    const int pair1             = pair0 + kBlock;
    const __nv_bfloat162 value0 = x[row_base + pair0];
    const __nv_bfloat162 value1 = x[row_base + pair1];
    const float2 x0             = __bfloat1622float2(value0);
    const float2 x1             = __bfloat1622float2(value1);
    const float local_sum       = x0.x * x0.x + x0.y * x0.y + x1.x * x1.x + x1.y * x1.y;

    __shared__ float warp_sums[kBlock / common::kWarpSize];
    __shared__ float inv_shared;
    const float sum = common::block_reduce_sum<kBlock>(local_sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(sum * (1.0f / 2048.0f) + eps); }
    __syncthreads();
    const float inv = inv_shared;

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
