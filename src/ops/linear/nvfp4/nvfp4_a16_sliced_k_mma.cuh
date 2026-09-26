#pragma once

// NVFP4 codes multiplied by their raw E4M3 G16 scales are exactly representable
// in BF16. The global weight divisor is applied to the complete FP32 reduction.
// Activations remain the represented public BF16 inputs.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/nvfp4/nvfp4_operands.h"
#include "ops/linear/nvfp4/nvfp4_shared.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Schedule, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a16_sliced_k_mma_kernel(
    Nvfp4A16Operands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_offset) {
    const auto* __restrict__ x             = operands.x;
    const auto* __restrict__ weight_codes  = operands.codes;
    const auto* __restrict__ weight_scales = operands.scales;
    const int kHidden                      = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    constexpr int ActiveTokens =
        Schedule::kTokenCapacity ? Schedule::kTokenCapacity : Schedule::kBlockTokens;
    constexpr bool MaskedColumns = !Schedule::kExactTokens;
    constexpr int kTileK         = Schedule::kTileKPerWarp;
    constexpr int kWarps         = Schedule::kKWarps;
    constexpr int kBlockRows     = Schedule::kBlockRows;
    constexpr int kBlockK        = Schedule::kBlockK;
    const int kGroups            = kHidden / kBlockK;
    constexpr int kBlockTokens   = Schedule::kBlockTokens;
    constexpr int kTokenMmas     = kBlockTokens / 8;
    static_assert(ActiveTokens >= 1 && ActiveTokens <= kBlockTokens);
    static_assert((kWarps & 1) == 0);

    union SharedStorage {
        struct {
            std::uint8_t codes[Schedule::kStages][kBlockRows][kBlockK / 2];
            std::uint8_t scales[Schedule::kStages][kBlockRows][kBlockK / 16];
            __nv_bfloat16 activations[Schedule::kStages][kWarps][kBlockTokens * kTileK];
        } staging;

        float partial[kWarps * kTokenMmas * 32 * 4];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    auto& shared =
        *reinterpret_cast<SharedStorage*>(nvfp4_shared_storage<Schedule::kSharedBytes>());
    auto& code_shared  = shared.staging.codes;
    auto& scale_shared = shared.staging.scales;
    auto& x_shared     = shared.staging.activations;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * (kBlockRows / (RowPolicy::kPaired ? 2 : 1));
    const int token_begin = token_offset + static_cast<int>(blockIdx.y) * ActiveTokens;
    const int live_columns =
        MaskedColumns ? min(ActiveTokens, operands.tokens - token_begin) : ActiveTokens;

    const auto stage_activation = [&](int stage, int group_k0) {
        constexpr auto kActivationCache = Schedule::kActivationCache;
        constexpr bool kPadded     = Schedule::kActivationStage == Nvfp4ActivationStage::PaddedZero;
        constexpr int kStageTokens = kPadded ? kBlockTokens : ActiveTokens;
        constexpr int kItems       = kStageTokens * (kTileK / 8);
        for (int item = lane; item < kItems; item += 32) {
            const int token = item / (kTileK / 8);
            const int k8    = item - token * (kTileK / 8);
            auto* destination =
                &x_shared[stage][warp][token * kTileK + nvfp4_a16_shared_col_64(token, k8 * 8)];
            if constexpr (!MaskedColumns && (!kPadded || ActiveTokens == kBlockTokens)) {
                cp_async<16, kActivationCache>(
                    destination, x + static_cast<std::int64_t>(token_begin + token) * kHidden +
                                     group_k0 + warp * kTileK + k8 * 8);
            } else {
                const int source_token = token < live_columns ? token : 0;
                cp_async_zfill<16, kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(token_begin + source_token) * kHidden + group_k0 +
                        warp * kTileK + k8 * 8,
                    token < live_columns ? 16 : 0);
            }
        }
    };

    constexpr int kCodeChunks  = kBlockK / 32;
    constexpr int kCodeSwizzle = kCodeChunks < 8 ? kCodeChunks - 1 : 7;
    const auto stage_codes     = [&](int stage, int group_k0) {
#pragma unroll
        for (int ri = 0; ri < Schedule::kRowsPerLoaderWarp; ++ri) {
            const int row    = warp * Schedule::kRowsPerLoaderWarp + ri;
            const int parent = row_policy.weight_row(row0, row, operands.rows);
            for (int chunk = lane; chunk < kCodeChunks; chunk += 32) {
                cp_async<16, Schedule::kWeightCache>(
                    &code_shared[stage][row][(chunk ^ (row & kCodeSwizzle)) * 16],
                    weight_codes + static_cast<std::int64_t>(parent) * (kHidden / 2) +
                        group_k0 / 2 + chunk * 16);
            }
            for (int tile = lane; tile < kBlockK / 64; tile += 32) {
                cp_async<4>(&scale_shared[stage][row][tile * 4],
                            weight_scales +
                                nvfp4_scale_byte_offset(parent, group_k0 / 16 + tile * 4, kHidden));
            }
        }
    };

    const int b_row                   = lane & 7;
    const int b_k_offset              = ((lane >> 3) & 1) << 3;
    const int warp_k0                 = warp * kTileK;
    float accumulators[kTokenMmas][4] = {};

#pragma unroll
    for (int stage = 0; stage < Schedule::kStages; ++stage) {
        if (stage < kGroups) {
            stage_codes(stage, stage * kBlockK);
            stage_activation(stage, stage * kBlockK);
            cp_commit();
        }
    }
#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int stage = group_index % Schedule::kStages;
        if (group_index + Schedule::kStages - 1 < kGroups)
            cp_wait<Schedule::kStages - 1>();
        else
            cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            const int code_col   = k_step * 16 + lid * 2;
            const auto load_pair = [&](int row, int col) {
                const int byte   = (warp_k0 + col) / 2;
                const int offset = ((byte / 16) ^ (row & kCodeSwizzle)) * 16 + (byte & 15);
                return nvfp4_scaled_pair_bf16(code_shared[stage][row][offset],
                                              scale_shared[stage][row][(warp_k0 + col) / 16]);
            };
            const unsigned a0 = load_pair(gid, code_col);
            const unsigned a1 = load_pair(gid + 8, code_col);
            const unsigned a2 = load_pair(gid, code_col + 8);
            const unsigned a3 = load_pair(gid + 8, code_col + 8);
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                unsigned b0;
                unsigned b1;
                const int row = token_mma * 8 + b_row;
                ldmatrix_x2(
                    b0, b1,
                    smem_addr(
                        &x_shared[stage][warp][row * kTileK + nvfp4_a16_shared_col_64(
                                                                  row, k_step * 16 + b_k_offset)]));
                mma_bf16(accumulators[token_mma][0], accumulators[token_mma][1],
                         accumulators[token_mma][2], accumulators[token_mma][3], a0, a1, a2, a3, b0,
                         b1);
            }
        }

        __syncthreads();
        const int next = group_index + Schedule::kStages;
        if (next < kGroups) {
            stage_codes(stage, next * kBlockK);
            stage_activation(stage, next * kBlockK);
            cp_commit();
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    if ((warp & 1) != 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            store_vec(partial + ((warp * kTokenMmas + token_mma) * 32 + lane) * 4,
                      make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                  accumulators[token_mma][2], accumulators[token_mma][3]));
        }
    }
    __syncthreads();

    if ((warp & 1) == 0) {
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            const float4 partner =
                load_vec<float4>(partial + (((warp + 1) * kTokenMmas + token_mma) * 32 + lane) * 4);
            accumulators[token_mma][0] += partner.x;
            accumulators[token_mma][1] += partner.y;
            accumulators[token_mma][2] += partner.z;
            accumulators[token_mma][3] += partner.w;
            if (warp != 0) {
                store_vec(partial + ((warp * kTokenMmas + token_mma) * 32 + lane) * 4,
                          make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                      accumulators[token_mma][2], accumulators[token_mma][3]));
            }
        }
    }
    __syncthreads();

    if (warp == 0) {
        const auto destination =
            linear_output_tile<kBlockRows / (RowPolicy::kPaired ? 2 : 1)>(output, row0);
        const float top_scale = operands.alpha, bottom_scale = operands.alpha;

#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            float4 sum = make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                     accumulators[token_mma][2], accumulators[token_mma][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kTokenMmas + token_mma) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int local_token = token_mma * 8 + 2 * lid;
            const int token0      = token_begin + local_token;
            const int row_a       = row_policy.weight_row(row0, gid, operands.rows);
            const int row_b       = row_policy.weight_row(row0, gid + 8, operands.rows);
            if constexpr (RowPolicy::kPaired) {
                if (local_token < live_columns)
                    epilogue.apply_pair(destination, row_a, token0, sum.x * top_scale,
                                        sum.z * bottom_scale);
                if (local_token + 1 < live_columns)
                    epilogue.apply_pair(destination, row_a, token0 + 1, sum.y * top_scale,
                                        sum.w * bottom_scale);
            } else {
                if (local_token < live_columns) {
                    destination.store(row_a, token0,
                                      epilogue.apply(row_a, token0, sum.x * top_scale));
                    destination.store(row_b, token0,
                                      epilogue.apply(row_b, token0, sum.z * bottom_scale));
                }
                if (local_token + 1 < live_columns) {
                    destination.store(row_a, token0 + 1,
                                      epilogue.apply(row_a, token0 + 1, sum.y * top_scale));
                    destination.store(row_b, token0 + 1,
                                      epilogue.apply(row_b, token0 + 1, sum.w * bottom_scale));
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
