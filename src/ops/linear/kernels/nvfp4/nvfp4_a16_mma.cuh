#pragma once
#include "ops/common/cuda/pipeline.cuh"

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/quant/nvfp4.cuh"

// NVFP4 codes multiplied by their raw E4M3 G16 scales are exactly representable
// in BF16. The matrix coefficient is applied to the complete FP32 reduction.
// Activations remain the represented public BF16 inputs.

#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/kernels/nvfp4/nvfp4_operands.h"

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
    auto* weight_shared = reinterpret_cast<__nv_bfloat16*>(shared_raw);
    auto* activation_shared =
        reinterpret_cast<__nv_bfloat16*>(shared_raw + Schedule::kExpandedWeightBytes);
    auto* code_shared  = shared_raw + Schedule::kExpandedWeightBytes + Schedule::kActivationBytes;
    auto* scale_shared = code_shared + Schedule::kRawStages * Schedule::kCodeStageBytes;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsTokens;
    const int wn   = warp - wm * Schedule::kWarpsTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    int row_tile = static_cast<int>(blockIdx.x), token_tile = static_cast<int>(blockIdx.y);
    if constexpr (Schedule::kRasterGroupRows > 0) {
        const int linear     = token_tile * static_cast<int>(gridDim.x) + row_tile;
        const int group_span = Schedule::kRasterGroupRows * static_cast<int>(gridDim.y);
        const int first_row  = linear / group_span * Schedule::kRasterGroupRows;
        const int group_rows =
            min(Schedule::kRasterGroupRows, static_cast<int>(gridDim.x) - first_row);
        const int within = linear % group_span;
        row_tile         = first_row + within % group_rows;
        token_tile       = within / group_rows;
    }
    const int row_begin   = row_tile * (BM / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + token_tile * BN;

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
                                                   common::swizzle_xor_64(local_token, kk)];
            if constexpr (FullTokens) {
                common::cp_async<16, Schedule::kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token) * K + k_begin + kk);
            } else {
                const bool valid = token < tokens;
                common::cp_async_zfill<16, Schedule::kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(valid ? token : 0) * K + k_begin + kk,
                    valid ? 16 : 0);
            }
        }
    };

    const auto code_offset = [](int row, int byte) {
        const int linear = row * (BK / 2) + byte;
        // Preserve 16-byte copy alignment while giving the MMA's eight row
        // groups distinct shared-memory banks for their packed code words.
        if constexpr (Schedule::kSwizzleCodes) return linear ^ ((row & 7) * 16);
        return linear;
    };

    const auto stage_codes = [&](int k_tile) {
        const int k_begin = k_tile * BK;
        const int slot    = k_tile % Schedule::kRawStages;
        auto* codes       = code_shared + slot * Schedule::kCodeStageBytes;
        auto* scales      = scale_shared + slot * Schedule::kScaleStageBytes;
        for (int item = tid; item < BM * (BK / 32); item += THREADS) {
            const int row = item / (BK / 32), chunk = item % (BK / 32);
            const int parent = row_policy.weight_row(row_begin, row, M);
            common::cp_async<16, Schedule::kWeightCache>(
                codes + code_offset(row, chunk * 16),
                weight_codes + static_cast<std::int64_t>(parent) * (K / 2) + k_begin / 2 +
                    chunk * 16);
        }
        for (int item = tid; item < BM * (BK / 64); item += THREADS) {
            const int row = item / (BK / 64), tile = item % (BK / 64);
            const int parent = row_policy.weight_row(row_begin, row, M);
            common::cp_async<4>(scales + row * (BK / 16) + tile * 4,
                                weight_scales + common::nvfp4_scale_byte_offset(
                                                    parent, k_begin / 16 + tile * 4, K));
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
                    common::nvfp4_scaled_pair_bf16(code_shared[code_offset(row, col / 2)],
                                                   scale_shared[row * (BK / 16) + col / 16]);
                common::store_vec(weight_shared + row * BK + common::swizzle_xor_64(row, col),
                                  widened);
            }
        }
    };

    stage_activation(0, 0);
    stage_codes(0);
    common::cp_commit();

    const int kTiles = K / BK;
#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % Schedule::kActivationStages;
        common::cp_wait<0>();
        __syncthreads();

        if constexpr (!Schedule::kRegisterWeights) {
            widen_codes();
            __syncthreads();
        }

        const int next = k_tile + 1;
        if (next < kTiles) {
            if constexpr (Schedule::kActivationStages == 2) { stage_activation(next & 1, next); }
            // RegW reads raw codes throughout this MMA tile. The alternate compact slot
            // keeps this prefetch disjoint; the loop's CTA barrier retires the reused slot.
            stage_codes(next);
            common::cp_commit();
        }

        constexpr int kSlots =
            Schedule::kFragmentPipeline == Nvfp4MmaFragmentPipeline::PingPong ? 2 : 1;
        unsigned a_fragments[kSlots][MT][4];
        unsigned b_fragments[kSlots][NT][2];
        const auto* tile_codes =
            code_shared + (k_tile % Schedule::kRawStages) * Schedule::kCodeStageBytes;
        const auto* tile_scales =
            scale_shared + (k_tile % Schedule::kRawStages) * Schedule::kScaleStageBytes;
        const auto load_fragments = [&](int slot, int k_step) {
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
                if constexpr (Schedule::kRegisterWeights) {
                    const int row         = wm * WM + mma_row * 16 + gid;
                    const int col         = k_step * 16 + 2 * lid;
                    const unsigned scale0 = tile_scales[row * (BK / 16) + k_step];
                    const unsigned scale1 = tile_scales[(row + 8) * (BK / 16) + k_step];
                    const unsigned sf0 = common::fp8_e4m3x2_to_bf16x2_bits(scale0 | (scale0 << 8));
                    const unsigned sf1 = common::fp8_e4m3x2_to_bf16x2_bits(scale1 | (scale1 << 8));
                    a_fragments[slot][mma_row][0] =
                        common::nvfp4_pair_bf16_scaled(tile_codes[code_offset(row, col / 2)], sf0);
                    a_fragments[slot][mma_row][1] = common::nvfp4_pair_bf16_scaled(
                        tile_codes[code_offset(row + 8, col / 2)], sf1);
                    a_fragments[slot][mma_row][2] = common::nvfp4_pair_bf16_scaled(
                        tile_codes[code_offset(row, col / 2 + 4)], sf0);
                    a_fragments[slot][mma_row][3] = common::nvfp4_pair_bf16_scaled(
                        tile_codes[code_offset(row + 8, col / 2 + 4)], sf1);
                } else {
                    const int row = wm * WM + mma_row * 16 + a_row_offset;
                    const int col = k_step * 16 + a_col_offset;
                    common::ldmatrix_x4(
                        a_fragments[slot][mma_row][0], a_fragments[slot][mma_row][1],
                        a_fragments[slot][mma_row][2], a_fragments[slot][mma_row][3],
                        common::smem_addr(
                            &weight_shared[row * BK + common::swizzle_xor_64(row, col)]));
                }
            }
#pragma unroll
            for (int mma_token = 0; mma_token < NT; ++mma_token) {
                const int row = wn * WN + mma_token * 8 + b_inner_row;
                const int col = k_step * 16 + b_col_offset;
                common::ldmatrix_x2(
                    b_fragments[slot][mma_token][0], b_fragments[slot][mma_token][1],
                    common::smem_addr(&activation_shared[stage * BN * BK + row * BK +
                                                         common::swizzle_xor_64(row, col)]));
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
                    common::mma_bf16(
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
                common::cp_commit();
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
