#pragma once

// Contiguous BF16 x BF16 SIMT core with exact or runtime column extents.
//
// Each row group owns RowsPerWarp output rows. WarpsPerRow warps cover disjoint K slices, and
// every loaded weight pack updates all ActiveTokens before it is discarded. The output policy
// owns the semantic row/token mapping, allowing pure Linear and fused projection Ops to share the
// same computation body without a packed intermediate.

#include "ops/linear/bf16/bf16_gemv.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int Values>
struct Bf16SimtFloatPack {
    float values[Values];
};

template <int Values>
__device__ __forceinline__ Bf16SimtFloatPack<Values>
bf16_simt_decode_pack(const Bf16GemvPack<Values>& packed) {
    Bf16SimtFloatPack<Values> result;
#pragma unroll
    for (int pair = 0; pair < Values / 2; ++pair) {
        const float2 values         = bf16x2_bits_to_float2(packed.words[pair]);
        result.values[2 * pair]     = values.x;
        result.values[2 * pair + 1] = values.y;
    }
    return result;
}

template <int Values, int Chains>
__device__ __forceinline__ void bf16_simt_accumulate(const Bf16SimtFloatPack<Values>& weight,
                                                     const Bf16GemvPack<Values>& activation,
                                                     float (&accumulators)[Chains]) {
    constexpr int kChainMask = Chains - 1;
#pragma unroll
    for (int pair = 0; pair < Values / 2; ++pair) {
        const float2 x = bf16x2_bits_to_float2(activation.words[pair]);
        accumulators[(2 * pair) & kChainMask] =
            fmaf(weight.values[2 * pair], x.x, accumulators[(2 * pair) & kChainMask]);
        accumulators[(2 * pair + 1) & kChainMask] =
            fmaf(weight.values[2 * pair + 1], x.y, accumulators[(2 * pair + 1) & kChainMask]);
    }
}

template <class Schedule, int ActiveTokens>
struct Bf16SimtSharedStorage {
    static constexpr int kReductionWarps = Schedule::kWarpsPerRow > 1 ? Schedule::kWarpsPerRow : 1;
    float partials[Schedule::kRowGroupsPerCta][Schedule::kRowsPerWarp][ActiveTokens]
                  [kReductionWarps];
};

template <class Geometry, int ActiveTokens, class Schedule>
__device__ __forceinline__ void bf16_simt_accumulate_direct_phase(
    const __nv_bfloat16* __restrict__ x, int phase, int warp_in_row, int lane,
    const Bf16GemvPack<Schedule::kValuesPerLane> (&packed_weights)[Schedule::kRowsPerWarp],
    float (&accumulators)[Schedule::kRowsPerWarp][ActiveTokens][Schedule::kAccumulatorChains],
    int live_tokens = ActiveTokens) {
    using Pack = Bf16GemvPack<Schedule::kValuesPerLane>;
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        const auto weight_values = bf16_simt_decode_pack(packed_weights[local_row]);
#pragma unroll
        for (int token0 = 0; token0 < ActiveTokens; token0 += Schedule::kTokenBatch) {
            Pack activation[Schedule::kTokenBatch];
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenBatch; ++local_token) {
                const int token = token0 + local_token;
                if (token < ActiveTokens) {
                    activation[local_token] = load_bf16_activation_phase<Geometry, Schedule>(
                        x + static_cast<std::int64_t>(min(token, live_tokens - 1)) *
                                Geometry::kInputRows,
                        phase, warp_in_row, lane);
                }
            }
#pragma unroll
            for (int local_token = 0; local_token < Schedule::kTokenBatch; ++local_token) {
                const int token = token0 + local_token;
                if (token < ActiveTokens) {
                    bf16_simt_accumulate(weight_values, activation[local_token],
                                         accumulators[local_row][token]);
                }
            }
        }
    }
}

template <class Geometry, int ActiveTokens, class Schedule>
__device__ __forceinline__ void bf16_simt_compute_rows(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight, int row0,
    int warp_in_row, int lane,
    float (&accumulators)[Schedule::kRowsPerWarp][ActiveTokens][Schedule::kAccumulatorChains],
    int live_tokens = ActiveTokens) {
    constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * kWarpSize * Schedule::kValuesPerLane;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);
    constexpr int kPhases = Geometry::kInputRows / kValuesPerPhase;
    using Pack            = Bf16GemvPack<Schedule::kValuesPerLane>;
    const int phase0      = Schedule::kPhaseOrder == Bf16PhaseOrder::Sequential
                                ? 0
                                : ((row0 / Schedule::kRowsPerWarp) * Schedule::kPhaseStride) % kPhases;

    if constexpr (Schedule::kActivationAccess == Bf16SimtActivationAccess::WarpPacked) {
#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < kPhases; ++iteration) {
            int phase = phase0 + iteration;
            if (phase >= kPhases) { phase -= kPhases; }
            Pack activation[ActiveTokens];
#pragma unroll
            for (int token = 0; token < ActiveTokens; ++token) {
                activation[token] = load_bf16_activation_phase<Geometry, Schedule>(
                    x + static_cast<std::int64_t>(min(token, live_tokens - 1)) *
                            Geometry::kInputRows,
                    phase, warp_in_row, lane);
            }

#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const Pack packed_weight = load_bf16_weight_phase<Geometry, Schedule>(
                    weight, row0 + local_row, phase, warp_in_row, lane);
                const auto weight_values = bf16_simt_decode_pack(packed_weight);
#pragma unroll
                for (int token = 0; token < ActiveTokens; ++token) {
                    bf16_simt_accumulate(weight_values, activation[token],
                                         accumulators[local_row][token]);
                }
            }
        }
    } else if constexpr (Schedule::kPrefetchDepth == 1) {
#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < kPhases; ++iteration) {
            int phase = phase0 + iteration;
            if (phase >= kPhases) { phase -= kPhases; }
            Pack packed_weights[Schedule::kRowsPerWarp];
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                packed_weights[local_row] = load_bf16_weight_phase<Geometry, Schedule>(
                    weight, row0 + local_row, phase, warp_in_row, lane);
            }
            bf16_simt_accumulate_direct_phase<Geometry, ActiveTokens, Schedule>(
                x, phase, warp_in_row, lane, packed_weights, accumulators, live_tokens);
        }
    } else {
        int phase = phase0;
        Pack current_weights[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            current_weights[local_row] = load_bf16_weight_phase<Geometry, Schedule>(
                weight, row0 + local_row, phase, warp_in_row, lane);
        }

#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < kPhases; ++iteration) {
            Pack next_weights[Schedule::kRowsPerWarp];
            int next_phase = phase + 1;
            if (next_phase == kPhases) { next_phase = 0; }
            if (iteration + 1 < kPhases) {
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    next_weights[local_row] = load_bf16_weight_phase<Geometry, Schedule>(
                        weight, row0 + local_row, next_phase, warp_in_row, lane);
                }
            }
            bf16_simt_accumulate_direct_phase<Geometry, ActiveTokens, Schedule>(
                x, phase, warp_in_row, lane, current_weights, accumulators, live_tokens);
            if (iteration + 1 < kPhases) {
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    current_weights[local_row] = next_weights[local_row];
                }
                phase = next_phase;
            }
        }
    }
}

template <class Geometry, int ActiveTokens, class Schedule, class OutputPolicy,
          bool RuntimeColumns = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    OutputPolicy output, int columns = ActiveTokens) {
    const int live_tokens = RuntimeColumns ? columns : ActiveTokens;
    static_assert(ActiveTokens >= 2 && ActiveTokens <= 32);
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    __shared__ Bf16SimtSharedStorage<Schedule, ActiveTokens> shared;

    const int lane        = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp        = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row_group   = warp / Schedule::kWarpsPerRow;
    const int warp_in_row = warp % Schedule::kWarpsPerRow;
    const int cta_row0    = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int row0        = cta_row0 + row_group * Schedule::kRowsPerWarp;
    float accumulators[Schedule::kRowsPerWarp][ActiveTokens][Schedule::kAccumulatorChains] = {};

    bf16_simt_compute_rows<Geometry, ActiveTokens, Schedule>(x, weight, row0, warp_in_row, lane,
                                                             accumulators, live_tokens);

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
        for (int token = 0; token < ActiveTokens; ++token) {
            float total = 0.0F;
#pragma unroll
            for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                total += accumulators[local_row][token][chain];
            }
            total = warp_reduce_sum(total);
            if constexpr (Schedule::kWarpsPerRow == 1) {
                if (lane == 0 && token < live_tokens) {
                    output.store(row0 + local_row, token, total);
                }
            } else if (lane == 0) {
                shared.partials[row_group][local_row][token][warp_in_row] = total;
            }
        }
    }

    if constexpr (Schedule::kWarpsPerRow > 1) {
        __syncthreads();
        if (warp_in_row == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
#pragma unroll
                for (int token = 0; token < ActiveTokens; ++token) {
                    const float partial = lane < Schedule::kWarpsPerRow
                                              ? shared.partials[row_group][local_row][token][lane]
                                              : 0.0F;
                    const float total   = warp_reduce_sum(partial);
                    if (lane == 0 && token < live_tokens) {
                        output.store(row0 + local_row, token, total);
                    }
                }
            }
        }
    }
}

struct Bf16SimtContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token, float value) const {
        data[static_cast<std::int64_t>(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

} // namespace ninfer::ops::detail
