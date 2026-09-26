#pragma once

#include "ops/linear/bf16/bf16_schedule.cuh"
#include "ops/common/math.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int Values>
struct alignas(Values* static_cast<int>(sizeof(__nv_bfloat16))) Bf16GemvPack {
    static_assert(Values == 4 || Values == 8 || Values == 16);
    std::uint32_t words[Values / 2];
};

static_assert(sizeof(Bf16GemvPack<4>) == 8);
static_assert(sizeof(Bf16GemvPack<8>) == 16);
static_assert(sizeof(Bf16GemvPack<16>) == 32);

template <int Values>
__device__ __forceinline__ Bf16GemvPack<Values> load_bf16_pack(const __nv_bfloat16* pointer) {
    if constexpr (Values <= 8) {
        return load_vec<Bf16GemvPack<Values>>(pointer);
    } else {
        Bf16GemvPack<Values> result;
        const Bf16GemvPack<8> low  = load_vec<Bf16GemvPack<8>>(pointer);
        const Bf16GemvPack<8> high = load_vec<Bf16GemvPack<8>>(pointer + 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            result.words[pair]     = low.words[pair];
            result.words[pair + 4] = high.words[pair];
        }
        return result;
    }
}

template <Bf16WeightCache Cache, int Values>
__device__ __forceinline__ Bf16GemvPack<Values>
load_bf16_weight_pack(const __nv_bfloat16* pointer) {
    if constexpr (Cache == Bf16WeightCache::Default) {
        return load_bf16_pack<Values>(pointer);
    } else if constexpr (Values == 4) {
        uint2 bits;
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                     : "=r"(bits.x), "=r"(bits.y)
                     : "l"(pointer));
        return load_vec<Bf16GemvPack<Values>>(&bits);
    } else if constexpr (Values == 8) {
        uint4 bits;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                     : "l"(pointer));
        return load_vec<Bf16GemvPack<Values>>(&bits);
    } else {
        Bf16GemvPack<Values> result;
        const Bf16GemvPack<8> low  = load_bf16_weight_pack<Cache, 8>(pointer);
        const Bf16GemvPack<8> high = load_bf16_weight_pack<Cache, 8>(pointer + 8);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            result.words[pair]     = low.words[pair];
            result.words[pair + 4] = high.words[pair];
        }
        return result;
    }
}

template <int Values, int AccumulatorChains>
__device__ __forceinline__ void accumulate_bf16_packs(const Bf16GemvPack<Values>& weight,
                                                      const Bf16GemvPack<Values>& activation,
                                                      float (&accumulators)[AccumulatorChains]) {
#pragma unroll
    for (int pair = 0; pair < Values / 2; ++pair) {
        const float2 w           = bf16x2_bits_to_float2(weight.words[pair]);
        const float2 x           = bf16x2_bits_to_float2(activation.words[pair]);
        constexpr int kChainMask = AccumulatorChains - 1;
        accumulators[(2 * pair) & kChainMask] =
            fmaf(w.x, x.x, accumulators[(2 * pair) & kChainMask]);
        accumulators[(2 * pair + 1) & kChainMask] =
            fmaf(w.y, x.y, accumulators[(2 * pair + 1) & kChainMask]);
    }
}

template <class Schedule>
struct Bf16GemvSharedStorage {
    static constexpr int kReductionWarps = Schedule::kWarpsPerRow > 1 ? Schedule::kWarpsPerRow : 1;

    float partials[Schedule::kRowGroupsPerCta][Schedule::kRowsPerWarp][kReductionWarps];
};

template <class Schedule>
__device__ __forceinline__ std::int32_t bf16_phase_offset(int phase, int warp_in_row, int lane) {
    constexpr int kValuesPerWarp = kWarpSize * Schedule::kValuesPerLane;
    return phase * Schedule::kWarpsPerRow * kValuesPerWarp + warp_in_row * kValuesPerWarp +
           lane * Schedule::kValuesPerLane;
}

template <class Schedule>
__device__ __forceinline__ Bf16GemvPack<Schedule::kValuesPerLane>
load_bf16_activation_phase(const __nv_bfloat16* activation, int phase, int warp_in_row, int lane) {
    const int offset = bf16_phase_offset<Schedule>(phase, warp_in_row, lane);
    return load_bf16_pack<Schedule::kValuesPerLane>(activation + offset);
}

template <class Schedule>
__device__ __forceinline__ Bf16GemvPack<Schedule::kValuesPerLane>
load_bf16_weight_phase(const __nv_bfloat16* weight, int row, int phase, int warp_in_row, int lane,
                       int input_rows) {
    const int K      = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    const int offset = bf16_phase_offset<Schedule>(phase, warp_in_row, lane);
    return load_bf16_weight_pack<Schedule::kWeightCache, Schedule::kValuesPerLane>(
        weight + static_cast<std::int64_t>(row) * K + offset);
}

template <class Schedule>
__device__ __forceinline__ int bf16_phase_index(int iteration, int row0, int phases) {
    if constexpr (Schedule::kPhaseOrder == Bf16PhaseOrder::Sequential) {
        return iteration;
    } else {
        const int row_group = row0 / Schedule::kRowsPerWarp;
        int phase           = iteration + (row_group * Schedule::kPhaseStride) % phases;
        if (phase >= phases) { phase -= phases; }
        return phase;
    }
}

template <class Schedule>
__device__ __forceinline__ void
compute_bf16_gemv_rows(const __nv_bfloat16* activation, const __nv_bfloat16* weight, int row0,
                       int warp_in_row, int lane,
                       float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains],
                       int input_rows) {
    const int K                   = Schedule::kStaticK ? Schedule::kStaticK : input_rows;
    constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * kWarpSize * Schedule::kValuesPerLane;
    const int phases              = K / kValuesPerPhase;
    using Pack                    = Bf16GemvPack<Schedule::kValuesPerLane>;

    if constexpr (Schedule::kPrefetchDepth == 1) {
#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < phases; ++iteration) {
            const int phase = bf16_phase_index<Schedule>(iteration, row0, phases);
            const Pack x_values =
                load_bf16_activation_phase<Schedule>(activation, phase, warp_in_row, lane);
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const Pack w_values = load_bf16_weight_phase<Schedule>(weight, row0 + local_row,
                                                                       phase, warp_in_row, lane, K);
                accumulate_bf16_packs(w_values, x_values, accumulators[local_row]);
            }
        }
    } else {
        const int first_phase = bf16_phase_index<Schedule>(0, row0, phases);
        Pack current_x =
            load_bf16_activation_phase<Schedule>(activation, first_phase, warp_in_row, lane);
        Pack current_w[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            current_w[local_row] = load_bf16_weight_phase<Schedule>(
                weight, row0 + local_row, first_phase, warp_in_row, lane, K);
        }

#pragma unroll Schedule::kPhaseUnroll
        for (int iteration = 0; iteration < phases; ++iteration) {
            Pack next_x;
            Pack next_w[Schedule::kRowsPerWarp];
            if (iteration + 1 < phases) {
                const int next_phase = bf16_phase_index<Schedule>(iteration + 1, row0, phases);
                next_x =
                    load_bf16_activation_phase<Schedule>(activation, next_phase, warp_in_row, lane);
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    next_w[local_row] = load_bf16_weight_phase<Schedule>(
                        weight, row0 + local_row, next_phase, warp_in_row, lane, K);
                }
            }
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                accumulate_bf16_packs(current_w[local_row], current_x, accumulators[local_row]);
            }
            if (iteration + 1 < phases) {
                current_x = next_x;
#pragma unroll
                for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                    current_w[local_row] = next_w[local_row];
                }
            }
        }
    }
}

template <class Schedule, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void bf16_a16_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight, Output output,
    Epilogue epilogue, int input_rows) {
    const int K = Schedule::kStaticK ? Schedule::kStaticK : input_rows;

    __shared__ Bf16GemvSharedStorage<Schedule> shared;
    const __nv_bfloat16* activation = x;
    if constexpr (Schedule::kActivationAccess == Bf16ActivationAccess::Shared) {
        extern __shared__ __align__(16) unsigned char activation_raw[];
        auto* destination  = reinterpret_cast<Bf16GemvPack<8>*>(activation_raw);
        const auto* source = reinterpret_cast<const Bf16GemvPack<8>*>(x);
        for (int pack = threadIdx.x; pack < K / 8; pack += Schedule::kThreads)
            destination[pack] = source[pack];
        __syncthreads();
        activation = reinterpret_cast<const __nv_bfloat16*>(activation_raw);
    }

    const int lane         = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp         = static_cast<int>(threadIdx.x) / kWarpSize;
    const int row_group    = warp / Schedule::kWarpsPerRow;
    const int warp_in_row  = warp % Schedule::kWarpsPerRow;
    const int cta_row0     = static_cast<int>(blockIdx.x) * Schedule::kBlockRows;
    const int row0         = cta_row0 + row_group * Schedule::kRowsPerWarp;
    const auto destination = linear_output_tile<Schedule::kBlockRows>(output, cta_row0);
    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};

    compute_bf16_gemv_rows<Schedule>(activation, weight, row0, warp_in_row, lane, accumulators, K);

    float warp_totals[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        warp_totals[local_row] = warp_reduce_sum(total);
    }

    if constexpr (Schedule::kWarpsPerRow == 1) {
        if (lane == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const float values[]{warp_totals[local_row]};
                linear_finish_row(destination, epilogue, row0 + local_row, 0, values, 1);
            }
        }
    } else {
        if (lane == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                shared.partials[row_group][local_row][warp_in_row] = warp_totals[local_row];
            }
        }
        __syncthreads();
        if (warp_in_row == 0) {
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const float partial = lane < Schedule::kWarpsPerRow
                                          ? shared.partials[row_group][local_row][lane]
                                          : 0.0F;
                const float total   = warp_reduce_sum(partial);
                if (lane == 0) {
                    const float values[]{total};
                    linear_finish_row(destination, epilogue, row0 + local_row, 0, values, 1);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
