#pragma once

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int Values>
struct alignas(Values / 2) Nvfp4CodePack {
    static_assert(Values == 8 || Values == 16 || Values == 32);
    std::uint32_t words[Values / 8];
};

static_assert(sizeof(Nvfp4CodePack<8>) == 4);
static_assert(sizeof(Nvfp4CodePack<16>) == 8);
static_assert(sizeof(Nvfp4CodePack<32>) == 16);

template <Nvfp4CodeCache Cache, int Values>
__device__ __forceinline__ Nvfp4CodePack<Values> load_nvfp4_codes(const std::uint8_t* pointer) {
    if constexpr (Cache == Nvfp4CodeCache::Default) {
        return load_vec<Nvfp4CodePack<Values>>(pointer);
    } else if constexpr (Values == 8) {
        Nvfp4CodePack<Values> result;
        asm volatile("ld.global.cg.u32 %0, [%1];\n" : "=r"(result.words[0]) : "l"(pointer));
        return result;
    } else if constexpr (Values == 16) {
        Nvfp4CodePack<Values> result;
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];\n"
                     : "=r"(result.words[0]), "=r"(result.words[1])
                     : "l"(pointer));
        return result;
    } else {
        Nvfp4CodePack<Values> result;
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(result.words[0]), "=r"(result.words[1]), "=r"(result.words[2]),
                       "=r"(result.words[3])
                     : "l"(pointer));
        return result;
    }
}

template <class Geometry, class Schedule>
struct Nvfp4GemvSharedStorage {
    static constexpr int kRawScaleBytes = Schedule::kScaleAccess == Nvfp4ScaleAccess::StagedRaw
                                              ? Schedule::kRowsPerCta * Geometry::kGroupsPerRow
                                              : 16;
    alignas(16) std::uint8_t raw_scales[kRawScaleBytes];
};

template <class Geometry, class Schedule>
__device__ __forceinline__ void
stage_nvfp4_scales(const std::uint8_t* __restrict__ scales,
                   Nvfp4GemvSharedStorage<Geometry, Schedule>& shared, int m_tile, int rmod_base) {
    if constexpr (Schedule::kScaleAccess == Nvfp4ScaleAccess::StagedRaw) {
        constexpr int kQuartetsPerCta = Schedule::kRowsPerCta / 4;
        constexpr int kTasks          = Geometry::kScaleTilesPerRow * kQuartetsPerCta;
        for (int task = static_cast<int>(threadIdx.x); task < kTasks; task += Schedule::kThreads) {
            const int scale_tile = task / kQuartetsPerCta;
            const int quartet    = task - scale_tile * kQuartetsPerCta;
            const std::int64_t source_offset =
                static_cast<std::int64_t>(m_tile * Geometry::kScaleTilesPerRow + scale_tile) * 512 +
                static_cast<std::int64_t>(rmod_base + quartet) * 16;
            const uint4 packed           = load_vec<uint4>(scales + source_offset);
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int quartile = 0; quartile < 4; ++quartile) {
                const int local_row = quartet * 4 + quartile;
                auto* destination   = reinterpret_cast<std::uint32_t*>(
                    shared.raw_scales + local_row * Geometry::kGroupsPerRow + scale_tile * 4);
                *destination = words[quartile];
            }
        }
        __syncthreads();
    }
}

template <class Geometry>
__device__ __forceinline__ std::int64_t nvfp4_scale_offset(int parent_row, int group) {
    return nvfp4_scale_byte<Geometry::kScaleTilesPerRow>(parent_row, group);
}

template <class Geometry, class Schedule>
__device__ __forceinline__ std::uint32_t
load_staged_scale_word(const Nvfp4GemvSharedStorage<Geometry, Schedule>& shared, int local_row,
                       int phase, int lane) {
    constexpr int kSubgroupWidth  = 64 / Schedule::kValuesPerLane;
    constexpr int kGroupsPerPhase = (32 * Schedule::kValuesPerLane) / 16;
    const int subgroup_lane       = lane & (kSubgroupWidth - 1);
    const int group_base          = phase * kGroupsPerPhase + (lane / kSubgroupWidth) * 4;
    std::uint32_t word            = 0;
    if (subgroup_lane == 0) {
        word = *reinterpret_cast<const std::uint32_t*>(
            shared.raw_scales + local_row * Geometry::kGroupsPerRow + group_base);
    }
    return __shfl_sync(0xffffffffU, word, 0, kSubgroupWidth);
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void load_nvfp4_coefficients(
    const std::uint8_t* __restrict__ scales,
    const Nvfp4GemvSharedStorage<Geometry, Schedule>& shared, int parent_row, int local_row,
    int phase, int lane, float inverse_weight_divisor,
    float (&coefficients)[Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16]) {
    constexpr int kGroupsPerLane =
        Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16;
    constexpr int kLanesPerGroup =
        Schedule::kValuesPerLane < 16 ? 16 / Schedule::kValuesPerLane : 1;
    const int value_begin = phase * 32 * Schedule::kValuesPerLane + lane * Schedule::kValuesPerLane;
    const int group_begin = value_begin / 16;

    if constexpr (Schedule::kScaleAccess == Nvfp4ScaleAccess::StagedRaw) {
        constexpr int kSubgroupWidth = 64 / Schedule::kValuesPerLane;
        const int subgroup_lane      = lane & (kSubgroupWidth - 1);
        const int first_byte         = (subgroup_lane / kLanesPerGroup) * kGroupsPerLane;
        const std::uint32_t word =
            load_staged_scale_word<Geometry, Schedule>(shared, local_row, phase, lane);
#pragma unroll
        for (int group = 0; group < kGroupsPerLane; ++group) {
            const auto scale    = static_cast<std::uint8_t>(word >> (8 * (first_byte + group)));
            coefficients[group] = decode_nvfp4_e4m3(scale) * inverse_weight_divisor;
        }
    } else {
#pragma unroll
        for (int group = 0; group < kGroupsPerLane; ++group) {
            const std::uint8_t scale =
                scales[nvfp4_scale_offset<Geometry>(parent_row, group_begin + group)];
            coefficients[group] = decode_nvfp4_e4m3(scale) * inverse_weight_divisor;
        }
    }
}

template <class Geometry, class Schedule>
__device__ __forceinline__ void
compute_nvfp4_rows(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                   const std::uint8_t* __restrict__ scales,
                   const Nvfp4GemvSharedStorage<Geometry, Schedule>& shared,
                   float inverse_weight_divisor, const int (&parent_rows)[Schedule::kRowsPerWarp],
                   int flat_row0, int lane,
                   float (&accumulators)[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains]) {
    constexpr int kValuesPerPhase = 32 * Schedule::kValuesPerLane;
    constexpr int kPhases         = Geometry::kInputRows / kValuesPerPhase;
    constexpr int kGroupsPerLane =
        Schedule::kValuesPerLane < 16 ? 1 : Schedule::kValuesPerLane / 16;
    static_assert((Geometry::kInputRows % kValuesPerPhase) == 0);
    const auto* activation_pairs = reinterpret_cast<const std::uint32_t*>(x);

#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        float coefficients[Schedule::kRowsPerWarp][kGroupsPerLane];
        Nvfp4CodePack<Schedule::kValuesPerLane> row_codes[Schedule::kRowsPerWarp];
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            load_nvfp4_coefficients<Geometry, Schedule>(
                scales, shared, parent_rows[local_row], flat_row0 + local_row, phase, lane,
                inverse_weight_divisor, coefficients[local_row]);
            const std::int64_t code_offset =
                static_cast<std::int64_t>(parent_rows[local_row]) * Geometry::kCodeBytesPerRow +
                phase * (kValuesPerPhase / 2) + lane * (Schedule::kValuesPerLane / 2);
            row_codes[local_row] = load_nvfp4_codes<Schedule::kCodeCache, Schedule::kValuesPerLane>(
                codes + code_offset);
        }

#pragma unroll
        for (int pair = 0; pair < Schedule::kPairsPerLane; ++pair) {
            const int activation_index =
                phase * (kValuesPerPhase / 2) + lane * Schedule::kPairsPerLane + pair;
            const float2 activation = bf16x2_bits_to_float2(activation_pairs[activation_index]);
            const int group         = ((lane * Schedule::kValuesPerLane & 15) + pair * 2) / 16;
#pragma unroll
            for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
                const std::uint32_t word  = row_codes[local_row].words[pair / 4];
                const std::uint8_t packed = static_cast<std::uint8_t>(word >> (8 * (pair & 3)));
                const float2 code         = decode_nvfp4_e2m1x2(packed);
                const float coefficient   = coefficients[local_row][group];
                constexpr int kChainMask  = Schedule::kAccumulatorChains - 1;
                accumulators[local_row][(2 * pair) & kChainMask] =
                    fmaf(code.x * coefficient, activation.x,
                         accumulators[local_row][(2 * pair) & kChainMask]);
                accumulators[local_row][(2 * pair + 1) & kChainMask] =
                    fmaf(code.y * coefficient, activation.y,
                         accumulators[local_row][(2 * pair + 1) & kChainMask]);
            }
        }
    }
}

template <class Geometry, class Schedule, class Epilogue, class Output>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void nvfp4_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse_weight_divisor, Epilogue epilogue,
    Output output) {
    static_assert((Geometry::kOutputRows % 128) == 0);
    static_assert((Schedule::kRowsPerCta % 4) == 0);
    static_assert((128 % Schedule::kRowsPerCta) == 0);

    __shared__ Nvfp4GemvSharedStorage<Geometry, Schedule> shared;
    constexpr int kCtasPerM128 = 128 / Schedule::kRowsPerCta;
    const int m_tile           = static_cast<int>(blockIdx.x) / kCtasPerM128;
    const int cta_in_tile      = static_cast<int>(blockIdx.x) - m_tile * kCtasPerM128;
    const int rmod_base        = cta_in_tile * (Schedule::kRowsPerCta / 4);
    stage_nvfp4_scales<Geometry, Schedule>(scales, shared, m_tile, rmod_base);

    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int flat_row0 = warp * Schedule::kRowsPerWarp;
    int parent_rows[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        const int flat_row     = flat_row0 + local_row;
        const int rmod         = rmod_base + flat_row / 4;
        const int quartile     = flat_row & 3;
        parent_rows[local_row] = m_tile * 128 + rmod + quartile * 32;
    }

    float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
    compute_nvfp4_rows<Geometry, Schedule>(x, codes, scales, shared, inverse_weight_divisor,
                                           parent_rows, flat_row0, lane, accumulators);

#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        float total = 0.0F;
#pragma unroll
        for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
            total += accumulators[local_row][chain];
        }
        total = warp_reduce_sum(total);
        if (lane == 0) {
            const int parent_row = parent_rows[local_row];
            output.store(parent_row, 0, epilogue.apply(parent_row, 0, total));
        }
    }
}

} // namespace ninfer::ops::detail
