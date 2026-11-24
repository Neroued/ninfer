#pragma once

#include "ops/moe/kernels/cuda/routing/launch.h"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"

namespace ninfer::ops::detail::moe_cuda::routing {
using common::cp_async_zfill;
using common::cp_commit;
using common::cp_wait;
using common::smem_addr;
using common::mma_bf16;
using common::ldmatrix_x2;
using common::ldmatrix_x4;

template <class Shape, int Rows = 16>
__global__ __launch_bounds__(Rows == 16 ? 256 : 128,
                             2) void router_mma_kernel(const __nv_bfloat16* __restrict__ x,
                                                       const __nv_bfloat16* __restrict__ weight,
                                                       float* __restrict__ scores, int tokens) {
    static_assert(Rows == 16 || Rows == 64);
    constexpr int BM = Rows, BN = 64, BK = 64, Stages = 2, Threads = Rows == 16 ? 256 : 128;
    constexpr int WarpRows = Rows == 16 ? 1 : 2, WarpColumns = Threads / 32 / WarpRows;
    constexpr int MT = BM / (16 * WarpRows), NT = BN / (8 * WarpColumns);
    __shared__ __align__(16) __nv_bfloat16 Ws[Stages][BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Xs[Stages][BN * BK];

    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int row0   = static_cast<int>(blockIdx.x) * BM;
    const int token0 = static_cast<int>(blockIdx.y) * BN;

    const int wm = warp % WarpRows, wn = warp / WarpRows;
    float acc[MT][NT][4] = {};

    auto stage_inputs = [&](int stage, int kt) {
        const int k0 = kt * BK;
        for (int item = tid; item < BM * (BK / 8); item += Threads) {
            const int row = item / (BK / 8);
            const int k8  = item - row * (BK / 8);
            const int rr  = row0 + row;
            auto* dst     = &Ws[stage][row * BK + common::swizzle_xor_64(row, k8 * 8)];
            const auto* src =
                weight +
                static_cast<std::int64_t>(rr < Shape::control_rows ? rr : 0) * Shape::hidden + k0 +
                k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, rr < Shape::control_rows ? 16 : 0);
        }
        for (int item = tid; item < BN * (BK / 8); item += Threads) {
            const int col = item / (BK / 8);
            const int k8  = item - col * (BK / 8);
            const int tt  = token0 + col;
            auto* dst     = &Xs[stage][col * BK + common::swizzle_xor_64(col, k8 * 8)];
            const auto* src =
                x + static_cast<std::int64_t>(tt < tokens ? tt : 0) * Shape::hidden + k0 + k8 * 8;
            cp_async_zfill<16, Cache::cg>(dst, src, tt < tokens ? 16 : 0);
        }
    };

#pragma unroll
    for (int stage = 0; stage < Stages; ++stage) {
        stage_inputs(stage, stage);
        cp_commit();
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int kt = 0; kt < Shape::hidden / BK; ++kt) {
        const int stage = kt & 1;
        cp_wait<Stages - 1>();
        __syncthreads();

#pragma unroll
        for (int ki = 0; ki < BK / 16; ++ki) {
            unsigned af[MT][4];
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                const int row = (wm * MT + m) * 16 + a_rowoff, col = ki * 16 + a_coloff;
                ldmatrix_x4(af[m][0], af[m][1], af[m][2], af[m][3],
                            smem_addr(&Ws[stage][row * BK + common::swizzle_xor_64(row, col)]));
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                unsigned bf[2];
                const int row = (wn * NT + n) * 8 + b_rin, col = ki * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1],
                            smem_addr(&Xs[stage][row * BK + common::swizzle_xor_64(row, col)]));
#pragma unroll
                for (int m = 0; m < MT; ++m) {
                    mma_bf16(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3], af[m][0],
                             af[m][1], af[m][2], af[m][3], bf[0], bf[1]);
                }
            }
        }

        __syncthreads();
        const int next = kt + Stages;
        if (next < Shape::hidden / BK) { stage_inputs(stage, next); }
        cp_commit();
    }
    cp_wait<0>();

#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int row = row0 + (wm * MT + m) * 16 + (lane >> 2) + (v / 2) * 8;
                const int col = token0 + (wn * NT + n) * 8 + (lane & 3) * 2 + (v & 1);
                if (row < Shape::control_rows && col < tokens) {
                    scores[static_cast<std::int64_t>(col) * Shape::score_rows + row] = acc[m][n][v];
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail::moe_cuda::routing
