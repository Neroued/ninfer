#pragma once

#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::gdn_control {
inline constexpr int kHeads = 48, kHidden = 5120, kLogicalRows = 96;
inline constexpr int kSmallTMax = 8, kSmallTKSlice = 512, kSmallTSplits = 10;
inline constexpr int kSmallTRowsPerBlock = 4, kSmallTThreads = 128;

template <int TokenTile, int KSlice, int RowsPerBlock>
__global__ void small_partial_kernel(const __nv_bfloat16* __restrict__ x,
                                     const __nv_bfloat16* __restrict__ a_weight,
                                     const __nv_bfloat16* __restrict__ b_weight,
                                     float* __restrict__ partial, std::int32_t t) {
    static_assert(TokenTile == kSmallTMax, "small-T token tile is fixed to 8");
    static_assert(KSlice == kSmallTKSlice, "small-T K split is fixed to 512");
    static_assert(RowsPerBlock == kSmallTRowsPerBlock, "small-T rows/block mismatch");
    constexpr int kVecsPerCol = KSlice / 8;

    __shared__ __align__(16) __nv_bfloat16 x_sh[TokenTile][KSlice];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int token0 = static_cast<int>(blockIdx.z) * TokenTile;
    if (token0 >= t) { return; }
    const int ncols = min(TokenTile, t - token0);
    const int split = static_cast<int>(blockIdx.y);
    const int k0    = split * KSlice;

    auto* x_sh_v = reinterpret_cast<uint4*>(x_sh);
    for (int i = static_cast<int>(threadIdx.x); i < ncols * kVecsPerCol;
         i += static_cast<int>(blockDim.x)) {
        const int col                   = i / kVecsPerCol;
        const int vec                   = i - col * kVecsPerCol;
        x_sh_v[col * kVecsPerCol + vec] = common::load_vec<uint4>(
            x + static_cast<std::int64_t>(token0 + col) * kHidden + k0 + vec * 8);
    }
    __syncthreads();

    const int logical_row = static_cast<int>(blockIdx.x) * RowsPerBlock + warp;
    if (logical_row >= kLogicalRows) { return; }
    const bool is_b = logical_row >= kHeads;
    const int row   = is_b ? logical_row - kHeads : logical_row;
    const __nv_bfloat16* wrow =
        (is_b ? b_weight : a_weight) + static_cast<std::int64_t>(row) * kHidden + k0;

    float acc[TokenTile];
#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) { acc[tt] = 0.0f; }

    for (int vec = lane; vec < kVecsPerCol; vec += 32) {
        const uint4 wv   = common::load_vec<uint4>(wrow + vec * 8);
        const float2 wf0 = common::bf16x2_bits_to_float2(wv.x);
        const float2 wf1 = common::bf16x2_bits_to_float2(wv.y);
        const float2 wf2 = common::bf16x2_bits_to_float2(wv.z);
        const float2 wf3 = common::bf16x2_bits_to_float2(wv.w);

#pragma unroll
        for (int tt = 0; tt < TokenTile; ++tt) {
            if (tt < ncols) {
                const uint4 xv   = x_sh_v[tt * kVecsPerCol + vec];
                const float2 xf0 = common::bf16x2_bits_to_float2(xv.x);
                const float2 xf1 = common::bf16x2_bits_to_float2(xv.y);
                const float2 xf2 = common::bf16x2_bits_to_float2(xv.z);
                const float2 xf3 = common::bf16x2_bits_to_float2(xv.w);
                acc[tt]          = fmaf(wf0.x, xf0.x, acc[tt]);
                acc[tt]          = fmaf(wf0.y, xf0.y, acc[tt]);
                acc[tt]          = fmaf(wf1.x, xf1.x, acc[tt]);
                acc[tt]          = fmaf(wf1.y, xf1.y, acc[tt]);
                acc[tt]          = fmaf(wf2.x, xf2.x, acc[tt]);
                acc[tt]          = fmaf(wf2.y, xf2.y, acc[tt]);
                acc[tt]          = fmaf(wf3.x, xf3.x, acc[tt]);
                acc[tt]          = fmaf(wf3.y, xf3.y, acc[tt]);
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) {
        if (tt < ncols) {
            float sum = common::warp_reduce_sum(acc[tt]);
            if (lane == 0) {
                const int token      = token0 + tt;
                partial[(static_cast<std::int64_t>(split) * t + token) * kLogicalRows +
                        logical_row] = sum;
            }
        }
    }
}

__global__ void small_reduce_kernel(const float* __restrict__ partial,
                                    const float* __restrict__ A_log,
                                    const float* __restrict__ dt_bias, float* __restrict__ g,
                                    float* __restrict__ beta, std::int32_t t) {
    const int i =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int elems = kHeads * t;
    if (i >= elems) { return; }

    const int row   = i % kHeads;
    const int token = i / kHeads;
    float acc_a     = 0.0f;
    float acc_b     = 0.0f;
#pragma unroll
    for (int split = 0; split < kSmallTSplits; ++split) {
        const std::int64_t base = (static_cast<std::int64_t>(split) * t + token) * kLogicalRows;
        acc_a += partial[base + row];
        acc_b += partial[base + kHeads + row];
    }

    const std::int64_t out_index = static_cast<std::int64_t>(token) * kHeads + row;
    const float sp               = common::softplus(acc_a + dt_bias[row]);
    g[out_index]                 = -expf(A_log[row]) * sp;
    beta[out_index]              = common::sigmoid(acc_b);
}

} // namespace ninfer::ops::detail::gdn_control
