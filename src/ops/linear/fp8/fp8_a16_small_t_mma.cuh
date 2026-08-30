#pragma once

// Small-T row-scaled E4M3 weight x BF16 activation Tensor Core mainloop.
//
// A CTA owns sixteen output rows and splits K across compile-time-selected warps. Persistent E4M3
// codes are widened exactly to BF16 MMA operands; the represented BF16 row multiplier is applied
// once to the complete FP32 dot product. The public activation is never quantized.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Geometry, int ActiveTokens, class Schedule, class Output = Fp8ContiguousOutput>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_a16_small_t_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Output output) {
    constexpr int kHidden     = Geometry::kInputRows;
    constexpr int kTileK      = Schedule::kTileKPerWarp;
    constexpr int kWarps      = Schedule::kKWarps;
    constexpr int kRowsPerCta = Schedule::kRowsPerCta;
    constexpr int kGroupK     = Schedule::kGroupK;
    constexpr int kGroups     = kHidden / kGroupK;
    constexpr int kTileTokens = Schedule::kTileTokens;
    constexpr int kTokenMmas  = kTileTokens / 8;
    static_assert((kHidden % kGroupK) == 0);
    static_assert((Geometry::kOutputRows % kRowsPerCta) == 0);
    static_assert(ActiveTokens >= 1 && ActiveTokens <= kTileTokens);
    static_assert((kWarps & 1) == 0);
    constexpr unsigned kMask = 0xffffffffU;

    union SharedStorage {
        struct {
            std::uint8_t codes[kRowsPerCta][kGroupK];
            __nv_bfloat16 activations[kWarps][kTileTokens * kTileK];
        } staging;

        float partial[kWarps * kTokenMmas * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared = shared.staging.codes;
    auto& x_shared    = shared.staging.activations;

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_activation = [&](int group_k0) {
        constexpr auto kActivationCache =
            Schedule::kActivationCache == Fp8A16SmallTMmaCache::Default ? Cache::ca : Cache::cg;
        constexpr bool kPadded =
            Schedule::kActivationStage == Fp8A16SmallTMmaActivationStage::PaddedZero;
        constexpr int kStageTokens = kPadded ? kTileTokens : ActiveTokens;
        constexpr int kItems       = kStageTokens * (kTileK / 8);
        for (int item = lane; item < kItems; item += 32) {
            const int token = item / (kTileK / 8);
            const int k8    = item - token * (kTileK / 8);
            auto* destination =
                &x_shared[warp][token * kTileK + fp8_a16_shared_col_64(token, k8 * 8)];
            if constexpr (!kPadded || ActiveTokens == kTileTokens) {
                cp_async<16, kActivationCache>(destination,
                                               x + static_cast<std::int64_t>(token) * kHidden +
                                                   group_k0 + warp * kTileK + k8 * 8);
            } else {
                const int source_token = token < ActiveTokens ? token : 0;
                cp_async_zfill<16, kActivationCache>(
                    destination,
                    x + static_cast<std::int64_t>(source_token) * kHidden + group_k0 +
                        warp * kTileK + k8 * 8,
                    token < ActiveTokens ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int group_k0) {
        constexpr auto kWeightCache =
            Schedule::kWeightCache == Fp8A16SmallTMmaCache::Default ? Cache::ca : Cache::cg;
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row = warp * Schedule::kRowsPerLoaderWarp + row_item;
            for (int chunk = lane; chunk < kGroupK / 16; chunk += 32) {
                const int swizzled_chunk = chunk ^ (row & 7);
                cp_async<16, kWeightCache>(&code_shared[row][swizzled_chunk * 16],
                                           weight_codes +
                                               static_cast<std::int64_t>(row0 + row) * kHidden +
                                               group_k0 + chunk * 16);
            }
        }
    };

    const int b_row                   = lane & 7;
    const int b_k_offset              = ((lane >> 3) & 1) << 3;
    const int warp_k0                 = warp * kTileK;
    float accumulators[kTokenMmas][4] = {};

    stage_codes(0);
    stage_activation(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            const int code_col        = k_step * 16 + lid * 2;
            const auto load_code_pair = [&](int row, int col) {
                const int chunk  = (warp_k0 + col) >> 4;
                const int offset = (chunk ^ (row & 7)) * 16 + (col & 15);
                return static_cast<unsigned>(
                    *reinterpret_cast<const std::uint16_t*>(&code_shared[row][offset]));
            };
            const unsigned a0 = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(gid, code_col));
            const unsigned a1 = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(gid + 8, code_col));
            const unsigned a2 = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(gid, code_col + 8));
            const unsigned a3 = fp8_e4m3x2_to_bf16x2_bits(load_code_pair(gid + 8, code_col + 8));
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                unsigned b0;
                unsigned b1;
                const int row = token_mma * 8 + b_row;
                ldmatrix_x2(
                    b0, b1,
                    smem_addr(&x_shared[warp][row * kTileK + fp8_a16_shared_col_64(
                                                                 row, k_step * 16 + b_k_offset)]));
                mma_bf16(accumulators[token_mma][0], accumulators[token_mma][1],
                         accumulators[token_mma][2], accumulators[token_mma][3], a0, a1, a2, a3, b0,
                         b1);
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            const int next_k0 = (group_index + 1) * kGroupK;
            stage_codes(next_k0);
            stage_activation(next_k0);
            cp_commit();
            cp_wait<0>();
            __syncthreads();
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
        unsigned lane_scale = 0;
        if (lid < 2) {
            lane_scale = static_cast<unsigned>(
                reinterpret_cast<const std::uint16_t*>(row_scales)[row0 + gid + lid * 8]);
        }
        const unsigned top_scale_bits    = __shfl_sync(kMask, lane_scale, lane & ~3);
        const unsigned bottom_scale_bits = __shfl_sync(kMask, lane_scale, (lane & ~3) + 1);
        const float top_scale =
            __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(top_scale_bits)));
        const float bottom_scale =
            __bfloat162float(__ushort_as_bfloat16(static_cast<std::uint16_t>(bottom_scale_bits)));

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
            const int token0 = token_mma * 8 + 2 * lid;
            if (token0 < ActiveTokens) {
                output.store(row0 + gid, token0, sum.x * top_scale);
                output.store(row0 + gid + 8, token0, sum.z * bottom_scale);
            }
            if (token0 + 1 < ActiveTokens) {
                output.store(row0 + gid, token0 + 1, sum.y * top_scale);
                output.store(row0 + gid + 8, token0 + 1, sum.w * bottom_scale);
            }
        }
    }
}

} // namespace ninfer::ops::detail
