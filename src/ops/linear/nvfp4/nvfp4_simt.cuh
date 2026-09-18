#pragma once

#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Nvfp4SimtFinalization {
    Elementwise,
    RowVector,
};

template <class Geometry, int ActiveTokens, class Schedule>
struct Nvfp4SimtSharedStorage {
    static constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane;
    static constexpr int kActivationElements =
        Schedule::kActivationAccess == Nvfp4SimtActivationAccess::SharedPhase
            ? Schedule::kTokenTile * kValuesPerPhase
            : 8;
    static constexpr int kPartialTokens = Schedule::kWarpsPerRow > 1 ? Schedule::kTokenTile : 1;

    Nvfp4GemvSharedStorage<Geometry, Schedule> gemv;
    alignas(16) __nv_bfloat16 activation[kActivationElements];
    float partials[Schedule::kRowGroupsPerCta][Schedule::kRowsPerWarp][kPartialTokens]
                  [Schedule::kWarpsPerRow];
};

template <int Values>
struct Nvfp4ActivationPack {
    static_assert(Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 2];
};

template <int Values>
__device__ __forceinline__ Nvfp4ActivationPack<Values>
load_nvfp4_activation_pack(const __nv_bfloat16* pointer) {
    Nvfp4ActivationPack<Values> result;
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
__device__ __forceinline__ void compute_nvfp4_simt_rows(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    Nvfp4SimtSharedStorage<Geometry, ActiveTokens, Schedule>& shared, float inverse_weight_divisor,
    const int (&parent_rows)[Schedule::kRowsPerWarp], int flat_row0, int token0, int warp_in_row,
    int lane,
    float (
        &accumulators)[Schedule::kRowsPerWarp][Schedule::kTokenTile][Schedule::kAccumulatorChains],
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
            constexpr int kStagePacks    = Schedule::kTokenTile * kPacksPerToken;
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
            Nvfp4ActivationPack<Schedule::kValuesPerLane> activation[Schedule::kTokenTile];
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
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
                for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
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
                for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
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

template <class Geometry, int ActiveTokens, class Schedule, class Epilogue, class OutputPolicy,
          Nvfp4SimtFinalization Finalization = Nvfp4SimtFinalization::Elementwise,
          bool RuntimeColumns                = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse_weight_divisor, Epilogue epilogue,
    OutputPolicy output, int columns = ActiveTokens) {
    const int live_tokens = RuntimeColumns ? columns : ActiveTokens;
    static_assert(!RuntimeColumns || Finalization == Nvfp4SimtFinalization::Elementwise);
    static_assert(ActiveTokens >= 2);
    static_assert(Schedule::kTokenTile <= ActiveTokens);
    static_assert((Geometry::kOutputRows % 128) == 0);
    static_assert((Schedule::kRowsPerCta % 4) == 0);
    static_assert((128 % Schedule::kRowsPerCta) == 0);

    constexpr int kRowBlocks  = Geometry::kOutputRows / Schedule::kRowsPerCta;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    const int linear_block    = static_cast<int>(blockIdx.x);
    int row_block;
    int token_tile;
    if constexpr (Schedule::kBlockOrder == Nvfp4SimtBlockOrder::RowsContiguous) {
        token_tile = linear_block / kRowBlocks;
        row_block  = linear_block - token_tile * kRowBlocks;
    } else {
        row_block  = linear_block / kTokenTiles;
        token_tile = linear_block - row_block * kTokenTiles;
    }
    const int token0 = kTokenTiles == 1 ? 0 : token_tile * Schedule::kTokenTile;

    __shared__ Nvfp4SimtSharedStorage<Geometry, ActiveTokens, Schedule> shared;
    constexpr int kCtasPerM128 = 128 / Schedule::kRowsPerCta;
    const int m_tile           = row_block / kCtasPerM128;
    const int cta_in_tile      = row_block - m_tile * kCtasPerM128;
    const int rmod_base        = cta_in_tile * (Schedule::kRowsPerCta / 4);
    stage_nvfp4_scales<Geometry, Schedule>(scales, shared.gemv, m_tile, rmod_base);

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int row_group   = warp / Schedule::kWarpsPerRow;
    const int warp_in_row = warp - row_group * Schedule::kWarpsPerRow;
    const int flat_row0   = row_group * Schedule::kRowsPerWarp;
    int parent_rows[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        const int flat_row     = flat_row0 + local_row;
        const int rmod         = rmod_base + flat_row / 4;
        const int quartile     = flat_row & 3;
        parent_rows[local_row] = m_tile * 128 + rmod + quartile * 32;
    }

    float accumulators[Schedule::kRowsPerWarp][Schedule::kTokenTile][Schedule::kAccumulatorChains] =
        {};
    compute_nvfp4_simt_rows<Geometry, ActiveTokens, Schedule>(
        x, codes, scales, shared, inverse_weight_divisor, parent_rows, flat_row0, token0,
        warp_in_row, lane, accumulators, live_tokens);

    if constexpr (Finalization == Nvfp4SimtFinalization::RowVector) {
        static_assert(Schedule::kTokenTile == ActiveTokens,
                      "row-vector finalization requires one CTA to own the full token row");
        if constexpr (Schedule::kWarpsPerRow == 1) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                float projected[ActiveTokens];
#pragma unroll
                for (int local_token = 0; local_token < ActiveTokens; ++local_token) {
                    float total = 0.0F;
#pragma unroll
                    for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                        total += accumulators[local_row][local_token][chain];
                    }
                    total = warp_reduce_sum(total);
                    if (lane == 0) {
                        const int parent_row   = parent_rows[local_row];
                        projected[local_token] = epilogue.apply(parent_row, local_token, total);
                    }
                }
                if (lane == 0) { output.store_row(parent_rows[local_row], projected); }
            }
        } else {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
                for (int local_token = 0; local_token < ActiveTokens; ++local_token) {
                    float total = 0.0F;
#pragma unroll
                    for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                        total += accumulators[local_row][local_token][chain];
                    }
                    total = warp_reduce_sum(total);
                    if (lane == 0) {
                        shared.partials[row_group][local_row][local_token][warp_in_row] = total;
                    }
                }
            }
            __syncthreads();
            if (warp_in_row == 0) {
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    float projected[ActiveTokens];
#pragma unroll
                    for (int local_token = 0; local_token < ActiveTokens; ++local_token) {
                        const float partial =
                            lane < Schedule::kWarpsPerRow
                                ? shared.partials[row_group][local_row][local_token][lane]
                                : 0.0F;
                        const float total = warp_reduce_sum(partial);
                        if (lane == 0) {
                            const int parent_row   = parent_rows[local_row];
                            projected[local_token] = epilogue.apply(parent_row, local_token, total);
                        }
                    }
                    if (lane == 0) { output.store_row(parent_rows[local_row], projected); }
                }
            }
        }
    } else {
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                const int token = token0 + local_token;
                if (token < live_tokens) {
                    float total = 0.0F;
#pragma unroll
                    for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                        total += accumulators[local_row][local_token][chain];
                    }
                    total = warp_reduce_sum(total);
                    if constexpr (Schedule::kWarpsPerRow == 1) {
                        if (lane == 0) {
                            const int parent_row = parent_rows[local_row];
                            output.store(parent_row, token,
                                         epilogue.apply(parent_row, token, total));
                        }
                    } else if (lane == 0) {
                        shared.partials[row_group][local_row][local_token][warp_in_row] = total;
                    }
                }
            }
        }

        if constexpr (Schedule::kWarpsPerRow > 1) {
            __syncthreads();
            if (warp_in_row == 0) {
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
                    for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                        const int token = token0 + local_token;
                        if (token < live_tokens) {
                            const float partial =
                                lane < Schedule::kWarpsPerRow
                                    ? shared.partials[row_group][local_row][local_token][lane]
                                    : 0.0F;
                            const float total = warp_reduce_sum(partial);
                            if (lane == 0) {
                                const int parent_row = parent_rows[local_row];
                                output.store(parent_row, token,
                                             epilogue.apply(parent_row, token, total));
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
