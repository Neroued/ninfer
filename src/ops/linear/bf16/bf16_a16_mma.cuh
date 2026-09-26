#pragma once
// BF16 operands remain represented exactly; the complete dot product accumulates in FP32.
#include "ops/linear/bf16/bf16_mma_common.cuh"

namespace ninfer::ops::detail {
template <class Schedule, bool FullTokens, class Output, class Epilogue, int Splits = 1>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight, Output output,
    Epilogue epilogue, int rows, int input_rows, int token_offset, int count) {
    const int M           = rows;
    const int K           = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tokens      = token_offset + count;
    const int split_k     = K / Splits;
    const int split_begin = static_cast<int>(blockIdx.z) * split_k;
    constexpr int BM      = Schedule::kBlockRows;
    constexpr int BN      = Schedule::kBlockTokens;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int MT      = Schedule::kMmaRows;
    constexpr int NT      = Schedule::kMmaTokens;
    constexpr int S       = Schedule::kStages;
    constexpr int THREADS = Schedule::kThreads;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* As = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* Bs = As + S * BM * BK;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int tiles_m = M / BM;
    const int tiles_n = count / BN + static_cast<int>(count % BN != 0);
    int tile_m        = 0;
    int tile_n        = 0;
    bf16_mma_tile_coordinates<Schedule>(static_cast<int>(blockIdx.x), tiles_m, tiles_n, tile_m,
                                        tile_n);
    const int m0 = tile_m * BM;
    const int n0 = token_offset + tile_n * BN;

    float accum[MT][NT][4] = {};

    auto stage_inputs = [&](int stage, int k_tile) {
        const int k0  = split_begin + k_tile * BK;
        auto* a_stage = As + stage * BM * BK;
        auto* b_stage = Bs + stage * BN * BK;

#pragma unroll 1
        for (int item = tid; item < BM * (BK / 8); item += THREADS) {
            const int row = item / (BK / 8);
            const int k8  = item - row * (BK / 8);
            const int kk  = k8 * 8;
            cp_async<16, Schedule::kWeightCache>(
                &a_stage[row * BK + bf16_mma_shared_col<Schedule>(row, kk)],
                &weight[static_cast<std::int64_t>(m0 + row) * K + k0 + kk]);
        }

#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int col   = item / (BK / 8);
            const int k8    = item - col * (BK / 8);
            const int kk    = k8 * 8;
            auto* dst       = &b_stage[col * BK + bf16_mma_shared_col<Schedule>(col, kk)];
            const int token = n0 + col;
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(token) * K + k0 + kk]);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(valid ? token : 0) * K + k0 + kk],
                    valid ? 16 : 0);
            }
        }
    };

    const int kTiles = split_k / BK;
#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        if (stage < kTiles) {
            stage_inputs(stage, stage);
            cp_commit();
        }
    }

#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= kTiles) {
            cp_wait<S - 1>();
        } else {
            // Once the producer stops refilling the ring, fewer than S groups remain. Drain them
            // instead of applying the steady-state wait count to the final consumer stage.
            cp_wait<0>();
        }
        __syncthreads();

        bf16_mma_compute_stage<Schedule>(As + stage * BM * BK, Bs + stage * BN * BK, accum, warp,
                                         lane);

        __syncthreads();
        const int next = k_tile + S;
        if (next < kTiles) {
            stage_inputs(stage, next);
            cp_commit();
        }
    }

    bf16_finish_mma_tile<Schedule, FullTokens>(output, epilogue, shared_raw, accum, m0, n0, M,
                                               tokens, warp, lane);
}
} // namespace ninfer::ops::detail
