#pragma once

#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/memory.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail::gdn_control {
template <int Heads, int Hidden>
__global__ void gemv_kernel(const __nv_bfloat16* x, const __nv_bfloat16* a_weight,
                            const __nv_bfloat16* b_weight, const float* A_log, const float* dt_bias,
                            float* g, float* beta) {
    constexpr int kHeads = Heads, kHidden = Hidden, kThreads = 256;
    const int global_row = static_cast<int>(blockIdx.x);
    const bool is_b      = global_row >= kHeads;
    const int row        = is_b ? global_row - kHeads : global_row;
    const auto* weight   = is_b ? b_weight : a_weight;
    __shared__ float warp_sums[kThreads / common::kWarpSize];

    float acc                   = 0.0f;
    constexpr int kPairs        = kHidden / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kHidden;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<std::int64_t>(row_base));
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
        const float2 wf = common::bf16x2_to_float2(w2[p]);
        const float2 xf = common::bf16x2_to_float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }

    acc = common::block_reduce_sum<kThreads>(acc, warp_sums);
    if (threadIdx.x == 0) {
        if (is_b) {
            beta[row] = common::sigmoid(acc);
        } else {
            const float sp = common::softplus(acc + dt_bias[row]);
            g[row]         = -expf(A_log[row]) * sp;
        }
    }
}

template <int Cols, int Threads>
__global__ void
tiled_gemv_kernel(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
                  const __nv_bfloat16* __restrict__ b_weight, const float* __restrict__ A_log,
                  const float* __restrict__ dt_bias, float* __restrict__ g,
                  float* __restrict__ beta, int t) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int logical_row = blockIdx.x, row = logical_row % 48;
    const bool is_b  = logical_row >= 48;
    const int token0 = blockIdx.y * Cols, ncols = min(Cols, t - token0);
    const auto* w   = (is_b ? b_weight : a_weight) + std::int64_t(row) * 2560;
    float acc[Cols] = {};
    __shared__ float warp_sums[Cols][Threads / 32];
    for (int vec = tid; vec < 2560 / 8; vec += Threads) {
        const uint4 wv     = common::load_vec<uint4>(w + vec * 8);
        const float2 wf[4] = {
            common::bf16x2_bits_to_float2(wv.x), common::bf16x2_bits_to_float2(wv.y),
            common::bf16x2_bits_to_float2(wv.z), common::bf16x2_bits_to_float2(wv.w)};
#pragma unroll
        for (int c = 0; c < Cols; ++c) {
            if (c < ncols) {
                const uint4 xv =
                    common::load_vec<uint4>(x + std::int64_t(token0 + c) * 2560 + vec * 8);
                const float2 xf[4] = {
                    common::bf16x2_bits_to_float2(xv.x), common::bf16x2_bits_to_float2(xv.y),
                    common::bf16x2_bits_to_float2(xv.z), common::bf16x2_bits_to_float2(xv.w)};
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    acc[c] = fmaf(wf[pair].x, xf[pair].x, acc[c]);
                    acc[c] = fmaf(wf[pair].y, xf[pair].y, acc[c]);
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < Cols; ++c) {
        const float sum = common::warp_reduce_sum(acc[c]);
        if (lane == 0) warp_sums[c][warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int c = 0; c < Cols; ++c) {
            if (c < ncols) {
                const float sum =
                    common::warp_reduce_sum(lane < Threads / 32 ? warp_sums[c][lane] : 0.0F);
                if (lane == 0) {
                    const std::int64_t out = std::int64_t(token0 + c) * 48 + row;
                    if (is_b)
                        beta[out] = common::sigmoid(sum);
                    else
                        g[out] = -expf(A_log[row]) * common::softplus(sum + dt_bias[row]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail::gdn_control
