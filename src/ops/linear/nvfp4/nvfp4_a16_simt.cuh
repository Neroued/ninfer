#pragma once

#include "ops/linear/nvfp4/nvfp4_a16_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Geometry, int ActiveTokens, class Schedule>
struct Nvfp4A16SimtSharedStorage {
    static constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane;
    static constexpr int kActivationElements =
        Schedule::kActivationAccess == Nvfp4SimtActivationAccess::SharedPhase
            ? Schedule::kBlockTokens * kValuesPerPhase
            : 8;
    static constexpr int kPartialTokens = Schedule::kWarpsPerRow > 1 ? Schedule::kBlockTokens : 1;

    Nvfp4A16GemvSharedStorage<Geometry, Schedule> gemv;
    alignas(16) __nv_bfloat16 activation[kActivationElements];
    float partials[Schedule::kRowGroupsPerCta][Schedule::kRowsPerWarp][kPartialTokens]
                  [Schedule::kWarpsPerRow];
};

template <int Values>
struct Nvfp4A16ActivationPack {
    static_assert(Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 2];
};

template <int Values>
__device__ __forceinline__ Nvfp4A16ActivationPack<Values>
load_nvfp4_activation_pack(const __nv_bfloat16* pointer) {
    Nvfp4A16ActivationPack<Values> result;
#pragma unroll
    for (int chunk = 0; chunk < Values / 8; ++chunk) {
        const uint4 packed          = load_vec<uint4>(pointer + chunk * 8);
        result.words[chunk * 4]     = packed.x;
        result.words[chunk * 4 + 1] = packed.y;
        result.words[chunk * 4 + 2] = packed.z;
        result.words[chunk * 4 + 3] = packed.w;
    }
    return result;
}

template <class Geometry, int ActiveTokens, class Schedule>
__device__ __forceinline__ void
compute_nvfp4_simt_rows(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                        const std::uint8_t* __restrict__ scales,
                        Nvfp4A16SimtSharedStorage<Geometry, ActiveTokens, Schedule>& shared,
                        float inverse_weight_divisor,
                        const int (&parent_rows)[Schedule::kRowsPerWarp], int flat_row0, int token0,
                        int warp_in_row, int lane,
                        float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kBlockTokens]
                                             [Schedule::kAccumulatorChains],
                        int live_tokens = ActiveTokens) {
    constexpr int kValuesPerWarpPhase = 32 * Schedule::kValuesPerLane;
    constexpr int kValuesPerPhase     = Schedule::kWarpsPerRow * kValuesPerWarpPhase;
    constexpr int kPhases             = Geometry::kInputRows / kValuesPerPhase;
    constexpr int kGroupsPerLane =
        Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);

#pragma unroll Schedule::kPhaseUnroll
    for (int phase = 0; phase < kPhases; ++phase) {
        if constexpr (Schedule::kActivationAccess == Nvfp4SimtActivationAccess::SharedPhase) {
            static_assert((kValuesPerPhase % 8) == 0);
            constexpr int kPacksPerToken = kValuesPerPhase / 8;
            constexpr int kStagePacks    = Schedule::kBlockTokens * kPacksPerToken;
            auto* destination            = reinterpret_cast<uint4*>(shared.activation);
            for (int task = static_cast<int>(threadIdx.x); task < kStagePacks;
                 task += Schedule::kThreads) {
                const int local_token = task / kPacksPerToken;
                const int local_pack  = task - local_token * kPacksPerToken;
                const int token       = token0 + local_token;
                if (token < ActiveTokens) {
                    const __nv_bfloat16* source =
                        x +
                        static_cast<std::int64_t>(min(token, live_tokens - 1)) *
                            Geometry::kInputRows +
                        phase * kValuesPerPhase + local_pack * 8;
                    destination[task] = load_vec<uint4>(source);
                }
            }
            __syncthreads();
        }

        const int warp_phase = phase * Schedule::kWarpsPerRow + warp_in_row;
        float coefficients[Schedule::kRowsPerWarp][kGroupsPerLane];
        Nvfp4CodePack<Schedule::kValuesPerLane> row_codes[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            load_nvfp4_coefficients<Geometry, Schedule>(
                scales, shared.gemv, parent_rows[local_row], flat_row0 + local_row, warp_phase,
                lane, inverse_weight_divisor, coefficients[local_row]);
            const std::int64_t code_offset =
                static_cast<std::int64_t>(parent_rows[local_row]) * Geometry::kCodeBytesPerRow +
                phase * (kValuesPerPhase / 2) + warp_in_row * (kValuesPerWarpPhase / 2) +
                lane * Schedule::kPairsPerLane;
            row_codes[local_row] = load_nvfp4_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(
                codes + code_offset);
        }

        if constexpr (Schedule::kActivationAccess == Nvfp4SimtActivationAccess::TokenPacked) {
            Nvfp4A16ActivationPack<Schedule::kValuesPerLane> activation[Schedule::kBlockTokens];
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                const int token = token0 + local_token;
                if (token < ActiveTokens) {
                    const int value_begin = phase * kValuesPerPhase +
                                            warp_in_row * kValuesPerWarpPhase +
                                            lane * Schedule::kValuesPerLane;
                    activation[local_token] = load_nvfp4_activation_pack<Schedule::kValuesPerLane>(
                        x +
                        static_cast<std::int64_t>(min(token, live_tokens - 1)) *
                            Geometry::kInputRows +
                        value_begin);
                }
            }

#pragma unroll
            for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
                float2 row_weight[Schedule::kRowsPerWarp];
                const int group = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                    const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                    const float2 code         = decode_nvfp4_e2m1x2(packed);
                    const float coefficient   = coefficients[local_row][group];
                    row_weight[local_row] = make_float2(code.x * coefficient, code.y * coefficient);
                }
#pragma unroll
                for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                    const int token = token0 + local_token;
                    if (token < ActiveTokens) {
                        const float2 activation_value =
                            bf16x2_bits_to_float2(activation[local_token].words[pair]);
                        constexpr int kChainMask = Schedule::kAccumulatorChains - 1;
#pragma unroll
                        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                            accumulators[local_row][local_token][(2 * pair) & kChainMask] =
                                fmaf(row_weight[local_row].x, activation_value.x,
                                     accumulators[local_row][local_token][(2 * pair) & kChainMask]);
                            accumulators[local_row][local_token][(2 * pair + 1) & kChainMask] =
                                fmaf(row_weight[local_row].y, activation_value.y,
                                     accumulators[local_row][local_token]
                                                 [(2 * pair + 1) & kChainMask]);
                        }
                    }
                }
            }
        } else {
#pragma unroll
            for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
                float2 row_weight[Schedule::kRowsPerWarp];
                const int group = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                    const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                    const float2 code         = decode_nvfp4_e2m1x2(packed);
                    const float coefficient   = coefficients[local_row][group];
                    row_weight[local_row] = make_float2(code.x * coefficient, code.y * coefficient);
                }
                const int pair_index = phase * (kValuesPerPhase / 2) +
                                       warp_in_row * (kValuesPerWarpPhase / 2) +
                                       lane * Schedule::kPairsPerLane + pair;
#pragma unroll
                for (int local_token = 0; local_token < Schedule::kBlockTokens; ++local_token) {
                    const int token = token0 + local_token;
                    if (token < ActiveTokens) {
                        float2 activation_value;
                        if constexpr (Schedule::kActivationAccess ==
                                      Nvfp4SimtActivationAccess::SharedPhase) {
                            const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(
                                shared.activation + local_token * kValuesPerPhase);
                            const int local_pair = warp_in_row * (kValuesPerWarpPhase / 2) +
                                                   lane * Schedule::kPairsPerLane + pair;
                            activation_value = bf16x2_bits_to_float2(activation_pairs[local_pair]);
                        } else {
                            const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(
                                x + static_cast<std::int64_t>(min(token, live_tokens - 1)) *
                                        Geometry::kInputRows);
                            activation_value = bf16x2_bits_to_float2(activation_pairs[pair_index]);
                        }
                        constexpr int kChainMask = Schedule::kAccumulatorChains - 1;
#pragma unroll
                        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                            accumulators[local_row][local_token][(2 * pair) & kChainMask] =
                                fmaf(row_weight[local_row].x, activation_value.x,
                                     accumulators[local_row][local_token][(2 * pair) & kChainMask]);
                            accumulators[local_row][local_token][(2 * pair + 1) & kChainMask] =
                                fmaf(row_weight[local_row].y, activation_value.y,
                                     accumulators[local_row][local_token]
                                                 [(2 * pair + 1) & kChainMask]);
                        }
                    }
                }
            }
        }

        if constexpr (Schedule::kActivationAccess == Nvfp4SimtActivationAccess::SharedPhase) {
            __syncthreads();
        }
    }
}

template <class Schedule, class Output, class Epilogue, class Rows>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_a16_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float alpha, Output output, Epilogue epilogue,
    Rows policy, int rows, int tokens) {
    using Geometry             = Nvfp4Geometry<128, Schedule::kStaticK>;
    constexpr int ActiveTokens = Schedule::kTokenCapacity;
    static_assert(ActiveTokens >= 1 && Schedule::kBlockTokens <= ActiveTokens);
    static_assert(Schedule::kBlockRows % 4 == 0 && 128 % Schedule::kBlockRows == 0);
    static_assert(!Rows::kPaired || Schedule::kRowsPerWarp % 2 == 0);
    const int live       = Schedule::kExactTokens ? ActiveTokens : tokens;
    const int row_blocks = rows / Schedule::kBlockRows;
    constexpr int token_tiles =
        (ActiveTokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const int block = blockIdx.x;
    int rb, tt;
    if constexpr (token_tiles == 1) {
        rb = block;
        tt = 0;
    } else if constexpr (Schedule::kBlockOrder == Nvfp4SimtBlockOrder::RowsContiguous) {
        tt = block / row_blocks;
        rb = block % row_blocks;
    } else {
        rb = block / token_tiles;
        tt = block % token_tiles;
    }
    const int token0 = tt * Schedule::kBlockTokens;
    __shared__ Nvfp4A16SimtSharedStorage<Geometry, ActiveTokens, Schedule> shared;
    nvfp4_stage_a16_scales<Geometry, Schedule>(scales, shared.gemv, rb, rows, policy);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int rg = warp / Schedule::kWarpsPerRow, wr = warp % Schedule::kWarpsPerRow;
    const int local0 = rg * Schedule::kRowsPerWarp;
    int parent[Schedule::kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < Schedule::kRowsPerWarp; ++r)
        parent[r] = nvfp4_a16_parent_row<Schedule>(rb, local0 + r, rows, policy);
    float acc[Schedule::kRowsPerWarp][Schedule::kBlockTokens][Schedule::kAccumulatorChains] = {};
    compute_nvfp4_simt_rows<Geometry, ActiveTokens, Schedule>(
        x, codes, scales, shared, alpha, parent, local0, token0, wr, lane, acc, live);
    float totals[Schedule::kRowsPerWarp][Schedule::kBlockTokens];
#pragma unroll
    for (int r = 0; r < Schedule::kRowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < Schedule::kBlockTokens; ++t) {
            float sum = 0;
#pragma unroll
            for (int c = 0; c < Schedule::kAccumulatorChains; ++c) sum += acc[r][t][c];
            sum = warp_reduce_sum(sum);
            if constexpr (Schedule::kWarpsPerRow == 1)
                totals[r][t] = sum;
            else if (lane == 0)
                shared.partials[rg][r][t][wr] = sum;
        }
    }
    if constexpr (Schedule::kWarpsPerRow > 1) {
        __syncthreads();
        if (wr == 0) {
#pragma unroll
            for (int r = 0; r < Schedule::kRowsPerWarp; ++r)
#pragma unroll
                for (int t = 0; t < Schedule::kBlockTokens; ++t) {
                    const float sum =
                        lane < Schedule::kWarpsPerRow ? shared.partials[rg][r][t][lane] : 0.0f;
                    totals[r][t] = warp_reduce_sum(sum);
                }
        }
    }
    if (wr == 0 && lane == 0) {
#pragma unroll
        for (int r = 0; r < Schedule::kRowsPerWarp; r += Rows::kPaired ? 2 : 1) {
            if constexpr (Rows::kPaired) {
#pragma unroll
                for (int t = 0; t < Schedule::kBlockTokens; ++t)
                    if (token0 + t < live)
                        epilogue.apply_pair(output, parent[r], token0 + t, totals[r][t],
                                            totals[r + 1][t]);
            } else
                linear_finish_row(output, epilogue, parent[r], token0, totals[r],
                                  min(Schedule::kBlockTokens, live - token0));
        }
    }
}
} // namespace ninfer::ops::detail
