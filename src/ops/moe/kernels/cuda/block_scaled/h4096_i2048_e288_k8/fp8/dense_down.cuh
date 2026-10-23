#pragma once

#include "ops/moe/kernels/cuda/block_scaled/h4096_i2048_e288_k8/fp8/compute.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::fp8 {

// A CTA owns each output tile across all eight experts, eliminating inter-expert
// atomic updates. G128 weight/input scales and route coefficients stay in FP32.
template <int Columns>
struct DenseDownStorage {
    TileStorage<Schedule<Columns, false>> tile;
    float coefficients[Columns];
};

template <int Columns>
__global__ void dense_eight_down(const std::uint8_t* input, const float* input_scales,
                                 ExpertMatrixDeviceView bank, Groups groups, const int* ids,
                                 const float* coefficients, float* output, int tokens,
                                 int row_window) {
    if (!dense_eight(groups, tokens)) return;
    using S = Schedule<Columns, false>;
    extern __shared__ __align__(16) unsigned char bytes[];
    auto& storage  = *reinterpret_cast<DenseDownStorage<Columns>*>(bytes);
    auto& tile     = storage.tile;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % S::warp_rows, wn = warp / S::warp_rows;
    const int column_tiles = (tokens + Columns - 1) / Columns;
    for (int flat = blockIdx.x; flat < (kHidden / 128) * column_tiles; flat += gridDim.x) {
        const int window = row_window, span = column_tiles * window;
        const int row0           = (flat / span * window + flat % window) * 128;
        const int token0         = (flat % span / window) * Columns;
        const int active_columns = min(Columns, ((tokens - token0 + 7) / 8) * 8);
        float acc[2][S::mma_columns][4]{};
        for (int expert_slot = 0; expert_slot < kTopK; ++expert_slot) {
            const int expert  = ids[expert_slot];
            const auto weight = resolve(bank, expert);
            if (threadIdx.x < Columns) {
                const int token = token0 + threadIdx.x;
                int assignment  = -1;
                if (token < tokens) {
                    for (int rank = 0; rank < kTopK; ++rank)
                        if (ids[token * kTopK + rank] == expert) assignment = token * kTopK + rank;
                }
                tile.sources[threadIdx.x] = assignment >= 0 ? groups.inverse[assignment] : -1;
                storage.coefficients[threadIdx.x] = assignment >= 0 ? coefficients[assignment] : 0;
            }
            __syncthreads();
            stage_inputs<S, false>(tile, 0, 0, row0, weight, input, active_columns);
            stage_inputs<S, false>(tile, 1, 1, row0, weight, input, active_columns);
            for (int kt = 0; kt < kIntermediate / 128; ++kt) {
                const int stage = kt % 2;
                cp_wait<1>();
                __syncthreads();
                const float sw =
                    __bfloat162float(weight.scales[(row0 / 128) * (kIntermediate / 128) + kt]);
#pragma unroll
                for (int n0 = 0; n0 < S::mma_columns; n0 += 2) {
                    const int first = wn * S::warp_columns + n0 * 8;
                    if (token0 + first < tokens) {
                        float p[2][2][4]{};
#pragma unroll
                        for (int ki = 0; ki < 4; ++ki) {
                            unsigned a[2][4];
#pragma unroll
                            for (int m = 0; m < 2; ++m) {
                                const int row =
                                              wm * 32 + m * 16 + (lane & 7) + ((lane >> 3 & 1) * 8),
                                          col = ki * 32 + (lane >> 4) * 16;
                                ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                                            smem_addr(tile.stage.weights[stage] + row * 128 +
                                                      swizzle<S>(row, col)));
                            }
#pragma unroll
                            for (int n = 0; n < 2; ++n) {
                                const int column = first + n * 8;
                                if (token0 + column < tokens) {
                                    unsigned b[2];
                                    const int row = column + (lane & 7),
                                              col = ki * 32 + ((lane >> 3) & 1) * 16;
                                    ldmatrix_x2(b[0], b[1],
                                                smem_addr(tile.stage.input[stage] + row * 128 +
                                                          swizzle<S>(row, col)));
#pragma unroll
                                    for (int m = 0; m < 2; ++m)
                                        mma_fp8_e4m3(p[m][n][0], p[m][n][1], p[m][n][2], p[m][n][3],
                                                     a[m][0], a[m][1], a[m][2], a[m][3], b[0],
                                                     b[1]);
                                }
                            }
                        }
#pragma unroll
                        for (int n = 0; n < 2; ++n)
                            if (token0 + first + n * 8 < tokens) {
#pragma unroll
                                for (int v = 0; v < 4; ++v) {
                                    const int column = first + n * 8 + (lane & 3) * 2 + (v & 1),
                                              source = tile.sources[column];
                                    const float scale =
                                        source >= 0 ? input_scales[std::int64_t(source) *
                                                                       (kIntermediate / 128) +
                                                                   kt] *
                                                          sw * storage.coefficients[column]
                                                    : 0;
                                    acc[0][n0 + n][v] = fmaf(p[0][n][v], scale, acc[0][n0 + n][v]);
                                    acc[1][n0 + n][v] = fmaf(p[1][n][v], scale, acc[1][n0 + n][v]);
                                }
                            }
                    }
                }
                __syncthreads();
                if (kt + 2 < kIntermediate / 128)
                    stage_inputs<S, false>(tile, stage, kt + 2, row0, weight, input,
                                           active_columns);
                else
                    cp_commit();
            }
            cp_wait<0>();
            __syncthreads();
        }
#pragma unroll
        for (int n = 0; n < S::mma_columns; ++n) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int token = token0 + wn * S::warp_columns + n * 8 + (lane & 3) * 2 + (v & 1);
                if (token < tokens) {
#pragma unroll
                    for (int m = 0; m < 2; ++m) {
                        const int row = row0 + wm * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                        output[std::int64_t(token) * kHidden + row] = acc[m][n][v];
                    }
                }
            }
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled::h4096_i2048_e288_k8::fp8
