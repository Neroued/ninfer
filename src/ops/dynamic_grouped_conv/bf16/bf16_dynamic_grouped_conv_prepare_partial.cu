#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_kernels.h"
#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/rowsplit_mma.cuh"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kHidden = 5120, kCoefficientRows = 1280, kBlockK = 64, kStages = 2;

__device__ __forceinline__ int shared_col(int row, int col) { return gemm_swz64(row, col); }

template <int Rows, int Columns>
struct Tile {
    static constexpr int column_warps = Rows == 16 ? Columns / 8 : 4;
    static constexpr int threads      = Rows / 16 * column_warps * 32;
    static constexpr int mmas         = Columns / column_warps / 8;
};

template <int Rows, int Columns, int SplitK>
__global__
__launch_bounds__(Tile<Rows, Columns>::threads, 1) void dynamic_grouped_conv_prepare_partial_kernel(
    const __nv_bfloat16* __restrict__ input, const __nv_bfloat16* __restrict__ weight,
    float* __restrict__ partial, int tokens) {
    constexpr int threads         = Tile<Rows, Columns>::threads;
    constexpr int mmas            = Tile<Rows, Columns>::mmas;
    constexpr int tiles_per_split = kHidden / kBlockK / SplitK;
    __shared__ __align__(16) __nv_bfloat16 ws[kStages][Rows * kBlockK];
    __shared__ __align__(16) __nv_bfloat16 xs[kStages][Columns * kBlockK];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, gid = lane / 4, lid = lane % 4;
    const int row0 = blockIdx.x * Rows, col0 = blockIdx.y * Columns, split = blockIdx.z;
    const int warp_row = warp / Tile<Rows, Columns>::column_warps;
    const int warp_col = warp % Tile<Rows, Columns>::column_warps;
    const int kt_begin = split * tiles_per_split;
    float acc[mmas][4] = {};
    auto stage_inputs  = [&](int stage, int kt) {
        for (int item = tid; item < Rows * (kBlockK / 8); item += threads) {
            int row = item / (kBlockK / 8), kk = item % (kBlockK / 8) * 8;
            cp_async<16, Cache::cg>(&ws[stage][row * kBlockK + shared_col(row, kk)],
                                     weight + static_cast<std::int64_t>(row0 + row) * kHidden +
                                         kt * kBlockK + kk);
        }
        for (int item = tid; item < Columns * (kBlockK / 8); item += threads) {
            int col = item / (kBlockK / 8), kk = item % (kBlockK / 8) * 8;
            const int safe_col = col0 + col < tokens ? col0 + col : 0;
            cp_async_zfill<16, Cache::cg>(&xs[stage][col * kBlockK + shared_col(col, kk)],
                                           input + static_cast<std::int64_t>(safe_col) * kHidden +
                                               kt * kBlockK + kk,
                                          col0 + col < tokens ? 16 : 0);
        }
    };
    for (int stage = 0; stage < kStages; ++stage) {
        stage_inputs(stage, kt_begin + stage);
        cp_commit();
    }
#pragma unroll 1
    for (int tile = 0; tile < tiles_per_split; ++tile) {
        const int stage = tile % kStages;
        if (tile + kStages <= tiles_per_split) {
            cp_wait<kStages - 1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
#pragma unroll
        for (int ki = 0; ki < kBlockK / 16; ++ki) {
            unsigned a[4];
            const int ar = warp_row * 16 + (lane & 7) + (((lane >> 3) & 1) * 8);
            const int ak = ki * 16 + (lane >> 4) * 8;
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        smem_addr(&ws[stage][ar * kBlockK + shared_col(ar, ak)]));
#pragma unroll
            for (int m = 0; m < mmas; ++m) {
                unsigned b[2];
                const int br = (warp_col * mmas + m) * 8 + (lane & 7);
                const int bk = ki * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(b[0], b[1], smem_addr(&xs[stage][br * kBlockK + shared_col(br, bk)]));
                mma_bf16(acc[m][0], acc[m][1], acc[m][2], acc[m][3], a[0], a[1], a[2], a[3], b[0],
                         b[1]);
            }
        }
        __syncthreads();
        if (tile + kStages < tiles_per_split) stage_inputs(stage, kt_begin + tile + kStages);
        cp_commit();
    }
#pragma unroll
    for (int m = 0; m < mmas; ++m) {
        const int row = row0 + warp_row * 16 + gid,
                  col = col0 + (warp_col * mmas + m) * 8 + 2 * lid;
        auto put      = [&](int r, int c, float value) {
            if (c < tokens)
                partial[(static_cast<std::int64_t>(split) * tokens + c) * kCoefficientRows + r] =
                    value;
        };
        put(row, col, acc[m][0]);
        put(row, col + 1, acc[m][1]);
        put(row + 8, col, acc[m][2]);
        put(row + 8, col + 1, acc[m][3]);
    }
}

template <int R, int C, int S>
void launch(const Tensor& input, const Weight& weight, float* partial, cudaStream_t stream) {
    const int tokens = input.ne[1] * input.ne[2];
    const dim3 grid(kCoefficientRows / R, (tokens + C - 1) / C, S);
    dynamic_grouped_conv_prepare_partial_kernel<R, C, S><<<grid, Tile<R, C>::threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input.data),
        static_cast<const __nv_bfloat16*>(weight.qdata), partial, tokens);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void bf16_dynamic_grouped_conv_prepare_partial_launch(DynamicConvPrepareRoute route,
                                                      const Tensor& input, const Weight& weight,
                                                      float* partial, cudaStream_t stream) {
    if (route.rows == 16 && route.split_k == 8) {
        switch (route.columns) {
#define COL(C)                                                                                     \
    case C:                                                                                        \
        return launch<16, C, 8>(input, weight, partial, stream)
            COL(8);
            COL(16);
            COL(32);
            COL(48);
#undef COL
        }
    }
    if (route.rows == 32 && route.split_k == 4) {
        if (route.columns == 32) return launch<32, 32, 4>(input, weight, partial, stream);
        if (route.columns == 64) return launch<32, 64, 4>(input, weight, partial, stream);
    }
    throw std::logic_error("dynamic grouped conv prepare: invalid production tile");
}
} // namespace ninfer::ops::detail
