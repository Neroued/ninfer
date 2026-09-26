#pragma once

// NVFP4 codes multiplied by their raw E4M3 G16 scales are exactly representable
// in BF16. The global weight divisor is applied to the complete FP32 reduction.
// Activations remain the represented public BF16 inputs.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Keep restricted pointers in the device ABI. Putting them in an aggregate loses
// NVCC alias information and increases register pressure in the K128 mainloop.
template <class Schedule, bool FullTokens, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a16_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales, float alpha, Output output, Epilogue epilogue,
    RowPolicy row_policy, int rows, int input_rows, int token_offset, int count) {
    const int M           = rows;
    const int K           = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int tokens      = token_offset + count;
    constexpr int BM      = Schedule::kBlockRows;
    constexpr int BN      = Schedule::kBlockTokens;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int WM      = Schedule::kWarpRows;
    constexpr int WN      = Schedule::kWarpTokens;
    constexpr int MT      = Schedule::kMmaRows;
    constexpr int NT      = Schedule::kMmaTokens;
    constexpr int KSUB    = Schedule::kMmaK;
    constexpr int THREADS = Schedule::kThreads;

    extern __shared__ __align__(16) unsigned char shared_raw[];
    auto* weight_shared     = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* activation_shared = weight_shared + BM * BK;
    auto* code_shared =
        reinterpret_cast<std::uint8_t*>(activation_shared + Schedule::kActivationStages * BN * BK);

    auto* scale_shared = code_shared + BM * (BK / 2);

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsTokens;
    const int wn   = warp - wm * Schedule::kWarpsTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int row_begin   = static_cast<int>(blockIdx.x) * (BM / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * BN;

    float accumulators[MT][NT][4] = {};

    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_col_offset = ((lane >> 3) & 1) << 3;

    const auto stage_activation = [&](int stage, int k_tile) {
        const int k_begin = k_tile * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int local_token = item / (BK / 8);
            const int k8          = item - local_token * (BK / 8);
            const int token       = token_begin + local_token;
            const int kk          = k8 * 8;
            auto* destination     = &activation_shared[stage * BN * BK + local_token * BK +
                                                   nvfp4_a16_shared_col_64(local_token, kk)];
            if constexpr (FullTokens) {
                cp_async<16, Schedule::kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token) * K + k_begin + kk);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(valid ? token : 0) * K + k_begin + kk,
                    valid ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int k_tile) {
        const int k_begin = k_tile * BK;
        for (int item = tid; item < BM * (BK / 32); item += THREADS) {
            const int row = item / (BK / 32), chunk = item % (BK / 32);
            const int parent = row_policy.weight_row(row_begin, row, M);
            cp_async<16, Schedule::kWeightCache>(code_shared + row * (BK / 2) + chunk * 16,
                                                 weight_codes +
                                                     static_cast<std::int64_t>(parent) * (K / 2) +
                                                     k_begin / 2 + chunk * 16);
        }
        for (int item = tid; item < BM * (BK / 64); item += THREADS) {
            const int row = item / (BK / 64), tile = item % (BK / 64);
            const int parent = row_policy.weight_row(row_begin, row, M);
            cp_async<4>(scale_shared + row * (BK / 16) + tile * 4,
                        weight_scales +
                            nvfp4_scale_byte_offset(parent, k_begin / 16 + tile * 4, K));
        }
    };
    const auto widen_codes = [&]() {
        const int half = lane >> 4, half_lane = lane & 15;
        for (int pair = warp * 2; pair < BM; pair += Schedule::kWarps * 2) {
            const int row = pair + half;
#pragma unroll
            for (int col32 = 0; col32 < BK; col32 += 32) {
                const int col = col32 + half_lane * 2;
                const unsigned widened =
                    nvfp4_scaled_pair_bf16(code_shared[row * (BK / 2) + col / 2],
                                           scale_shared[row * (BK / 16) + col / 16]);
                store_vec(weight_shared + row * BK + nvfp4_a16_shared_col_64(row, col), widened);
            }
        }
    };

    stage_activation(0, 0);
    stage_codes(0);
    cp_commit();

    const int kTiles = K / BK;
#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kActivationStages;
        cp_wait<0>();
        __syncthreads();

        widen_codes();
        __syncthreads();

        const int next = k_tile + 1;
        if (next < kTiles) {
            if constexpr (Schedule::kActivationStages == 2) { stage_activation(next & 1, next); }
            stage_codes(next);
            cp_commit();
        }

        constexpr int kSlots =
            Schedule::kFragmentPipeline == Nvfp4MmaFragmentPipeline::PingPong ? 2 : 1;
        unsigned a_fragments[kSlots][MT][4];
        unsigned b_fragments[kSlots][NT][2];
        const auto load_fragments = [&](int slot, int k_step) {
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
                const int row = wm * WM + mma_row * 16 + a_row_offset;
                const int col = k_step * 16 + a_col_offset;
                ldmatrix_x4(
                    a_fragments[slot][mma_row][0], a_fragments[slot][mma_row][1],
                    a_fragments[slot][mma_row][2], a_fragments[slot][mma_row][3],
                    smem_addr(&weight_shared[row * BK + nvfp4_a16_shared_col_64(row, col)]));
            }
#pragma unroll
            for (int mma_token = 0; mma_token < NT; ++mma_token) {
                const int row = wn * WN + mma_token * 8 + b_inner_row;
                const int col = k_step * 16 + b_col_offset;
                ldmatrix_x2(b_fragments[slot][mma_token][0], b_fragments[slot][mma_token][1],
                            smem_addr(&activation_shared[stage * BN * BK + row * BK +
                                                         nvfp4_a16_shared_col_64(row, col)]));
            }
        };

        load_fragments(0, 0);
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            const int slot = k_step % kSlots;
            if constexpr (kSlots == 2) {
                if (k_step + 1 < KSUB) load_fragments(slot ^ 1, k_step + 1);
            } else if (k_step != 0) {
                load_fragments(0, k_step);
            }
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
#pragma unroll
                for (int mma_token = 0; mma_token < NT; ++mma_token) {
                    mma_bf16(
                        accumulators[mma_row][mma_token][0], accumulators[mma_row][mma_token][1],
                        accumulators[mma_row][mma_token][2], accumulators[mma_row][mma_token][3],
                        a_fragments[slot][mma_row][0], a_fragments[slot][mma_row][1],
                        a_fragments[slot][mma_row][2], a_fragments[slot][mma_row][3],
                        b_fragments[slot][mma_token][0], b_fragments[slot][mma_token][1]);
                }
            }
        }

        if constexpr (Schedule::kActivationStages == 1) {
            if (next < kTiles) {
                __syncthreads();
                stage_activation(0, next);
                cp_commit();
            }
        }
    }

    const auto destination =
        linear_output_tile<BM / (RowPolicy::kPaired ? 2 : 1)>(output, row_begin);
    if constexpr (requires {
                      epilogue.template finish_tile<Schedule, FullTokens>(
                          destination, shared_raw, accumulators, row_begin, token_begin, M, tokens);
                  }) {
        // Complete the represented row scaling in FP32 before entering the fused operation.
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const float scale0 = alpha, scale1 = alpha;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                accumulators[mi][ni][0] *= scale0;
                accumulators[mi][ni][1] *= scale0;
                accumulators[mi][ni][2] *= scale1;
                accumulators[mi][ni][3] *= scale1;
            }
        }
        epilogue.template finish_tile<Schedule, FullTokens>(destination, shared_raw, accumulators,
                                                            row_begin, token_begin, M, tokens);
    } else {
        static_assert(!RowPolicy::kPaired, "paired rows require a collective epilogue");
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row0     = row_policy.weight_row(row_begin, wm * WM + mi * 16 + gid, M);
            const int row1     = row_policy.weight_row(row_begin, wm * WM + mi * 16 + gid + 8, M);
            const float scale0 = alpha, scale1 = alpha;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int token0 = token_begin + wn * WN + ni * 8 + 2 * lid;
                const auto& v    = accumulators[mi][ni];
                if (FullTokens || token0 < tokens) {
                    destination.store(row0, token0, epilogue.apply(row0, token0, v[0] * scale0));
                    destination.store(row1, token0, epilogue.apply(row1, token0, v[2] * scale1));
                }
                if (FullTokens || token0 + 1 < tokens) {
                    destination.store(row0, token0 + 1,
                                      epilogue.apply(row0, token0 + 1, v[1] * scale0));
                    destination.store(row1, token0 + 1,
                                      epilogue.apply(row1, token0 + 1, v[3] * scale1));
                }
            }
        }
    }
}
} // namespace ninfer::ops::detail
