#pragma once

// Reusable row-scaled FP8 CUDA-core mainloop for a compile-time number of BF16 activation
// columns. A CTA owns one row tile and one compile-time token tile, reusing each decoded weight
// pair across that token tile before advancing K. Output, epilogue, and row policies let fused
// consumers retain their observable semantics without duplicating the contraction.

#include "ops/linear/fp8/fp8_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Fp8SmallTFinalization : std::uint8_t {
    Elementwise,
    RowVector,
};

template <int Values>
struct Fp8ActivationPack {
    static_assert(Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 2];
};

template <int Values>
__device__ __forceinline__ Fp8ActivationPack<Values>
load_fp8_activation_pack(const __nv_bfloat16* pointer) {
    Fp8ActivationPack<Values> result;
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

template <class Schedule>
struct Fp8SmallTSharedStorage {
    static constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    static constexpr int kActivationElements =
        Schedule::kActivationAccess == Fp8SmallTActivationAccess::SharedPhase
            ? Schedule::kTokenTile * kValuesPerPhase
            : 8;
    alignas(16) __nv_bfloat16 activation[kActivationElements];
};

template <class Geometry, int ActiveTokens, class Schedule, class Output,
          class Epilogue = Fp8IdentityEpilogue, class RowPolicy = Fp8GemvIdentityRows,
          bool PairRows                      = false,
          Fp8SmallTFinalization Finalization = Fp8SmallTFinalization::Elementwise>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void fp8_small_t_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Output output, Epilogue epilogue = {},
    RowPolicy row_policy = {}) {
    static_assert(ActiveTokens >= 2);
    static_assert(Schedule::kTokenTile <= ActiveTokens);
    static_assert(!PairRows || (Schedule::kRowsPerWarp % 2) == 0);
    constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    constexpr int kPhases = Geometry::kInputRows / kValuesPerPhase;
    constexpr int kStoredRowsPerWarp =
        PairRows ? Schedule::kRowsPerWarp / 2 : Schedule::kRowsPerWarp;
    constexpr int kStoredRowsPerCta = Schedule::kWarpsPerCta * kStoredRowsPerWarp;
    constexpr int kRowBlocks        = Geometry::kOutputRows / Schedule::kRowsPerCta;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;

    const int linear_block = static_cast<int>(blockIdx.x);
    int row_block;
    int token_tile;
    if constexpr (Schedule::kBlockOrder == Fp8SmallTBlockOrder::RowsContiguous) {
        token_tile = linear_block / kRowBlocks;
        row_block  = linear_block - token_tile * kRowBlocks;
    } else {
        row_block  = linear_block / kTokenTiles;
        token_tile = linear_block - row_block * kTokenTiles;
    }
    const int token0 = kTokenTiles == 1 ? 0 : token_tile * Schedule::kTokenTile;

    __shared__ Fp8SmallTSharedStorage<Schedule> shared;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int row_begin = row_block * kStoredRowsPerCta + warp * kStoredRowsPerWarp;
    float accumulators[Schedule::kRowsPerWarp][Schedule::kTokenTile][Schedule::kAccumulatorChains] =
        {};

#pragma unroll Schedule::kPhaseUnroll
    for (int phase = 0; phase < kPhases; ++phase) {
        if constexpr (Schedule::kActivationAccess == Fp8SmallTActivationAccess::SharedPhase) {
            constexpr int kPacksPerToken = kValuesPerPhase / 8;
            constexpr int kPacks         = Schedule::kTokenTile * kPacksPerToken;
            auto* destination            = reinterpret_cast<uint4*>(shared.activation);
            for (int task = static_cast<int>(threadIdx.x); task < kPacks;
                 task += Schedule::kThreads) {
                const int local_token = task / kPacksPerToken;
                const int local_pack  = task - local_token * kPacksPerToken;
                const int token       = token0 + local_token;
                if (token < ActiveTokens) {
                    destination[task] = load_vec<uint4>(
                        x + static_cast<std::int64_t>(token) * Geometry::kInputRows +
                        phase * kValuesPerPhase + local_pack * 8);
                }
            }
        }

        const int value_begin = phase * kValuesPerPhase + lane * Schedule::kValuesPerLane;
        Fp8CodePack<Schedule::kValuesPerLane> row_codes[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int weight_row = row_policy.weight_row(row_begin, local_row);
            row_codes[local_row] = load_fp8_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(
                weight_codes + static_cast<std::int64_t>(weight_row) * Geometry::kInputRows +
                value_begin);
        }

        Fp8ActivationPack<Schedule::kValuesPerLane> activation[Schedule::kTokenTile];
        if constexpr (Schedule::kActivationAccess == Fp8SmallTActivationAccess::SharedPhase) {
            __syncthreads();
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                if (token0 + local_token < ActiveTokens) {
                    activation[local_token] = load_fp8_activation_pack<Schedule::kValuesPerLane>(
                        shared.activation + local_token * kValuesPerPhase +
                        lane * Schedule::kValuesPerLane);
                }
            }
        } else {
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                const int token = token0 + local_token;
                if (token < ActiveTokens) {
                    activation[local_token] = load_fp8_activation_pack<Schedule::kValuesPerLane>(
                        x + static_cast<std::int64_t>(token) * Geometry::kInputRows + value_begin);
                }
            }
        }

        constexpr int kChainMask = Schedule::kAccumulatorChains - 1;
#pragma unroll
        for (int pair = 0; pair < Schedule::kValuesPerLane / 2; ++pair) {
            float2 weights[Schedule::kRowsPerWarp];
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::uint32_t word   = row_codes[local_row].words[pair >> 1];
                const std::uint16_t packed = static_cast<std::uint16_t>(word >> ((pair & 1) * 16));
                weights[local_row]         = decode_fp8_e4m3x2(packed);
            }
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                if (token0 + local_token >= ActiveTokens) { continue; }
                const float2 value = bf16x2_bits_to_float2(activation[local_token].words[pair]);
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    accumulators[local_row][local_token][(2 * pair) & kChainMask] =
                        fmaf(weights[local_row].x, value.x,
                             accumulators[local_row][local_token][(2 * pair) & kChainMask]);
                    accumulators[local_row][local_token][(2 * pair + 1) & kChainMask] =
                        fmaf(weights[local_row].y, value.y,
                             accumulators[local_row][local_token][(2 * pair + 1) & kChainMask]);
                }
            }
        }

        if constexpr (Schedule::kActivationAccess == Fp8SmallTActivationAccess::SharedPhase) {
            __syncthreads();
        }
    }

    if constexpr (Finalization == Fp8SmallTFinalization::RowVector) {
        static_assert(!PairRows, "row-vector finalization does not pair output rows");
        static_assert(Schedule::kTokenTile == ActiveTokens,
                      "row-vector finalization requires one CTA to own the full token row");
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int parent_row = row_policy.weight_row(row_begin, local_row);
            const float scale    = __bfloat162float(row_scales[parent_row]);
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
                    projected[local_token] = epilogue.apply(parent_row, local_token, total * scale);
                }
            }
            if (lane == 0) { output.store_row(parent_row, projected); }
        }
    } else if constexpr (PairRows) {
#pragma unroll
        for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
            const int token = token0 + local_token;
            if (token >= ActiveTokens) { continue; }
            float totals[Schedule::kRowsPerWarp];
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                float total = 0.0F;
#pragma unroll
                for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                    total += accumulators[local_row][local_token][chain];
                }
                totals[local_row] = warp_reduce_sum(total);
            }
            if (lane == 0) {
#pragma unroll
                for (int local_row = 0; local_row < kStoredRowsPerWarp; ++local_row) {
                    const int first_row = row_policy.weight_row(row_begin, local_row);
                    const int second_row =
                        row_policy.weight_row(row_begin, kStoredRowsPerWarp + local_row);
                    const float first =
                        epilogue.apply(first_row, token,
                                       totals[local_row] * __bfloat162float(row_scales[first_row]));
                    const float second =
                        epilogue.apply(second_row, token,
                                       totals[kStoredRowsPerWarp + local_row] *
                                           __bfloat162float(row_scales[second_row]));
                    output.store_pair(row_begin + local_row, token, first, second);
                }
            }
        }
    } else {
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int parent_row = row_policy.weight_row(row_begin, local_row);
            const float scale    = __bfloat162float(row_scales[parent_row]);
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenTile; ++local_token) {
                const int token = token0 + local_token;
                if (token >= ActiveTokens) { continue; }
                float total = 0.0F;
#pragma unroll
                for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                    total += accumulators[local_row][local_token][chain];
                }
                total = warp_reduce_sum(total);
                if (lane == 0) {
                    output.store(parent_row, token,
                                 epilogue.apply(parent_row, token, total * scale));
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
