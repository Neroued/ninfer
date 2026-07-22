#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q4GemvActivationAccess {
    Direct,
    CtaSharedFullK,
};

enum class Q4GemvLaneMapping {
    PackedByte2,
    PackedWord8,
};

enum class Q4GemvDecodeMode {
    ScalarInteger,
    Fp16Mantissa,
};

enum class Q4GemvCodeTransfer {
    SyncVector16,
    AsyncVector16,
};

enum class Q4GemvScaleAccess {
    Scalar16Shuffle,
    SharedPair32,
};

template <int RowsPerCta, int WarpsPerRow, int GroupsPerWarpTile, int PipelineStages,
          Q4GemvActivationAccess ActivationAccess, Q4GemvLaneMapping LaneMapping,
          Q4GemvDecodeMode DecodeMode, Q4GemvCodeTransfer CodeTransfer,
          Q4GemvScaleAccess ScaleAccess, Cache CodeCache, int StaticGroupsPerRow,
          int LaunchBoundsMinBlocks>
struct Q4RowSplitGemvSchedule {
    static_assert(RowsPerCta > 0, "Q4 GEMV requires at least one row per CTA");
    static_assert(WarpsPerRow > 0, "Q4 GEMV requires at least one warp per row");
    static_assert(GroupsPerWarpTile > 0 && (GroupsPerWarpTile % 2) == 0,
                  "Q4 GEMV group tiles must contain whole scale pairs");
    static_assert(GroupsPerWarpTile <= 32,
                  "Q4 GEMV scalar-shuffle scale access is limited to one warp");
    static_assert(PipelineStages >= 1 && PipelineStages <= 8,
                  "Q4 GEMV pipeline depth must fit cp.async wait-group immediates");
    static_assert(StaticGroupsPerRow == 0 ||
                      (StaticGroupsPerRow > 0 && (StaticGroupsPerRow % 2) == 0),
                  "Q4 GEMV static row ownership must contain whole scale pairs");
    static_assert(LaunchBoundsMinBlocks >= 1, "Q4 GEMV launch-bounds occupancy must be positive");

    static constexpr int kRowsPerCta            = RowsPerCta;
    static constexpr int kWarpsPerRow           = WarpsPerRow;
    static constexpr int kGroupsPerWarpTile     = GroupsPerWarpTile;
    static constexpr int kPipelineStages        = PipelineStages;
    static constexpr auto kActivationAccess     = ActivationAccess;
    static constexpr auto kLaneMapping          = LaneMapping;
    static constexpr auto kDecodeMode           = DecodeMode;
    static constexpr auto kCodeTransfer         = CodeTransfer;
    static constexpr auto kScaleAccess          = ScaleAccess;
    static constexpr auto kCodeCache            = CodeCache;
    static constexpr int kStaticGroupsPerRow    = StaticGroupsPerRow;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks;

    static constexpr int kCtaWarps = kRowsPerCta * kWarpsPerRow;
    static constexpr int kThreads  = kCtaWarps * 32;
    static constexpr int kCodeVectorsPerTile =
        kGroupsPerWarpTile * Q4RowSplitStorage::kCodeBytesPerGroup / sizeof(uint4);
    static constexpr int kScalePairsPerTile = kGroupsPerWarpTile / 2;

    static_assert(kCtaWarps <= 32, "Q4 GEMV cannot exceed the CUDA CTA warp limit");
    static_assert(kThreads <= 1024, "Q4 GEMV cannot exceed the CUDA CTA thread limit");
    static_assert((kGroupsPerWarpTile * Q4RowSplitStorage::kCodeBytesPerGroup) % sizeof(uint4) == 0,
                  "Q4 GEMV code tile must be representable as 16-byte vectors");

    static_assert((kLaneMapping == Q4GemvLaneMapping::PackedByte2 &&
                   kDecodeMode == Q4GemvDecodeMode::ScalarInteger) ||
                      (kLaneMapping == Q4GemvLaneMapping::PackedWord8 &&
                       kDecodeMode == Q4GemvDecodeMode::Fp16Mantissa),
                  "Q4 GEMV lane mapping and decode mode must describe the same packed ownership");
    static_assert((kCodeTransfer == Q4GemvCodeTransfer::SyncVector16 &&
                   kScaleAccess == Q4GemvScaleAccess::Scalar16Shuffle && kPipelineStages == 1 &&
                   kCodeCache == Cache::ca) ||
                      (kCodeTransfer == Q4GemvCodeTransfer::AsyncVector16 &&
                       kScaleAccess == Q4GemvScaleAccess::SharedPair32),
                  "Q4 GEMV transfer and scale schedules must select an implemented path");
};

using Q4GemvR4W1DirectSchedule =
    Q4RowSplitGemvSchedule<4, 1, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedWord8, Q4GemvDecodeMode::Fp16Mantissa,
                           Q4GemvCodeTransfer::AsyncVector16, Q4GemvScaleAccess::SharedPair32,
                           Cache::ca, 0, 1>;
using Q4GemvR1W8DirectSchedule =
    Q4RowSplitGemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, 80, 1>;

template <class Schedule, Q4GemvScaleAccess ScaleAccess = Schedule::kScaleAccess>
struct Q4GemvTileStorage;

template <class Schedule>
struct Q4GemvTileStorage<Schedule, Q4GemvScaleAccess::Scalar16Shuffle> {
    __align__(16)
        uint4 codes[Schedule::kCtaWarps][Schedule::kPipelineStages][Schedule::kCodeVectorsPerTile];
};

template <class Schedule>
struct Q4GemvTileStorage<Schedule, Q4GemvScaleAccess::SharedPair32> {
    __align__(16)
        uint4 codes[Schedule::kCtaWarps][Schedule::kPipelineStages][Schedule::kCodeVectorsPerTile];
    __align__(16) std::uint32_t
        scale_pairs[Schedule::kCtaWarps][Schedule::kPipelineStages][Schedule::kScalePairsPerTile];
};

template <class Schedule>
__device__ __forceinline__ void q4_gemv_issue_async_tile(
    uint4* __restrict__ shared_codes, std::uint32_t* __restrict__ shared_scale_pairs,
    const std::uint8_t* __restrict__ code_row, const std::uint8_t* __restrict__ scale_row,
    int group_begin, int active_groups, int lane) {
    static_assert(Schedule::kCodeTransfer == Q4GemvCodeTransfer::AsyncVector16);
    static_assert(Schedule::kScaleAccess == Q4GemvScaleAccess::SharedPair32);

    const int active_code_vectors =
        active_groups * Q4RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
    const auto* global_codes = reinterpret_cast<const uint4*>(
        code_row + static_cast<std::int64_t>(group_begin) * Q4RowSplitStorage::kCodeBytesPerGroup);

    for (int vector = lane; vector < Schedule::kCodeVectorsPerTile; vector += 32) {
        if (vector < active_code_vectors) {
            cp_async<16, Schedule::kCodeCache>(&shared_codes[vector], &global_codes[vector]);
        } else {
            shared_codes[vector] = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    const int active_scale_pairs = active_groups / 2;
    const std::uint8_t* global_scale_pairs =
        scale_row + static_cast<std::int64_t>(group_begin) * Q4RowSplitStorage::kScaleBytesPerGroup;
    for (int pair = lane; pair < Schedule::kScalePairsPerTile; pair += 32) {
        if (pair < active_scale_pairs) {
            cp_async<4>(&shared_scale_pairs[pair], global_scale_pairs + pair * 4);
        } else {
            shared_scale_pairs[pair] = 0u;
        }
    }
    cp_commit();
}

template <class Schedule>
__device__ __forceinline__ float
q4_gemv_consume_word_tile(const uint4* __restrict__ shared_codes,
                          const std::uint32_t* __restrict__ shared_scale_pairs,
                          const __nv_bfloat16* __restrict__ activation, int group_begin,
                          int active_groups, int lane, float accumulator) {
    static_assert(Schedule::kLaneMapping == Q4GemvLaneMapping::PackedWord8);
    static_assert(Schedule::kDecodeMode == Q4GemvDecodeMode::Fp16Mantissa);

    const auto* packed_words = reinterpret_cast<const std::uint32_t*>(shared_codes);
    const int lane_group     = lane >> 3;
    const int lane_in_group  = lane & 7;

#pragma unroll
    for (int group_base = 0; group_base < Schedule::kGroupsPerWarpTile; group_base += 4) {
        const int local_group = group_base + lane_group;
        if (local_group < active_groups) {
            const std::uint32_t packed     = packed_words[group_base * 8 + lane];
            const std::uint32_t scale_pair = shared_scale_pairs[local_group >> 1];
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(scale_pair >> ((local_group & 1) * 16));

            float weights[8];
            Q4SimtDecodeAtom::decode_eight(packed, scale_bits, weights);

            const int k_begin =
                (group_begin + local_group) * Q4RowSplitStorage::kGroupK + lane_in_group * 8;
            const uint4 activation_bits = load_vec<uint4>(activation + k_begin);
            const float2 x0             = bf16x2_bits_to_float2(activation_bits.x);
            const float2 x1             = bf16x2_bits_to_float2(activation_bits.y);
            const float2 x2             = bf16x2_bits_to_float2(activation_bits.z);
            const float2 x3             = bf16x2_bits_to_float2(activation_bits.w);

            accumulator = fmaf(weights[0], x0.x, accumulator);
            accumulator = fmaf(weights[1], x0.y, accumulator);
            accumulator = fmaf(weights[2], x1.x, accumulator);
            accumulator = fmaf(weights[3], x1.y, accumulator);
            accumulator = fmaf(weights[4], x2.x, accumulator);
            accumulator = fmaf(weights[5], x2.y, accumulator);
            accumulator = fmaf(weights[6], x3.x, accumulator);
            accumulator = fmaf(weights[7], x3.y, accumulator);
        }
    }
    return accumulator;
}

template <class Schedule>
__device__ __forceinline__ float q4_gemv_dot_word_async(
    Q4GemvTileStorage<Schedule>& shared_tiles, int cta_warp,
    const __nv_bfloat16* __restrict__ activation, const std::uint8_t* __restrict__ code_row,
    const std::uint8_t* __restrict__ scale_row, int group_begin, int group_end, int lane) {
    constexpr int kGroupsPerTile    = Schedule::kGroupsPerWarpTile;
    constexpr int kPipelineStages   = Schedule::kPipelineStages;
    constexpr int kPipelinePrefetch = kPipelineStages - 1;

    const int tile_count = div_up(group_end - group_begin, kGroupsPerTile);
    float accumulator    = 0.0f;

#pragma unroll
    for (int prefetch = 0; prefetch < kPipelinePrefetch; ++prefetch) {
        if (prefetch < tile_count) {
            const int tile_group_begin = group_begin + prefetch * kGroupsPerTile;
            const int active_groups    = min(kGroupsPerTile, group_end - tile_group_begin);
            q4_gemv_issue_async_tile<Schedule>(shared_tiles.codes[cta_warp][prefetch],
                                               shared_tiles.scale_pairs[cta_warp][prefetch],
                                               code_row, scale_row, tile_group_begin, active_groups,
                                               lane);
        } else {
            cp_commit();
        }
    }

    for (int tile = 0; tile < tile_count; ++tile) {
        const int fetch = tile + kPipelinePrefetch;
        if (fetch < tile_count) {
            const int fetch_group_begin   = group_begin + fetch * kGroupsPerTile;
            const int fetch_active_groups = min(kGroupsPerTile, group_end - fetch_group_begin);
            const int fetch_stage         = fetch % kPipelineStages;
            q4_gemv_issue_async_tile<Schedule>(shared_tiles.codes[cta_warp][fetch_stage],
                                               shared_tiles.scale_pairs[cta_warp][fetch_stage],
                                               code_row, scale_row, fetch_group_begin,
                                               fetch_active_groups, lane);
        } else {
            cp_commit();
        }

        cp_wait<kPipelinePrefetch>();
        __syncwarp();

        const int tile_group_begin = group_begin + tile * kGroupsPerTile;
        const int active_groups    = min(kGroupsPerTile, group_end - tile_group_begin);
        const int consume_stage    = tile % kPipelineStages;
        accumulator                = q4_gemv_consume_word_tile<Schedule>(
            shared_tiles.codes[cta_warp][consume_stage],
            shared_tiles.scale_pairs[cta_warp][consume_stage], activation, tile_group_begin,
            active_groups, lane, accumulator);
        __syncwarp();
    }
    return accumulator;
}

template <class Schedule, int GroupsPerWarp>
__device__ __forceinline__ float
q4_gemv_dot_byte_static(Q4GemvTileStorage<Schedule>& shared_tiles, int cta_warp,
                        const __nv_bfloat16* __restrict__ activation,
                        const std::uint8_t* __restrict__ code_row,
                        const std::uint8_t* __restrict__ scale_row, int group_begin, int lane) {
    static_assert(Schedule::kLaneMapping == Q4GemvLaneMapping::PackedByte2);
    static_assert(Schedule::kDecodeMode == Q4GemvDecodeMode::ScalarInteger);
    static_assert(Schedule::kCodeTransfer == Q4GemvCodeTransfer::SyncVector16);
    static_assert(Schedule::kScaleAccess == Q4GemvScaleAccess::Scalar16Shuffle);
    static_assert(Schedule::kPipelineStages == 1);
    static_assert(GroupsPerWarp > 0 && GroupsPerWarp <= Schedule::kGroupsPerWarpTile);
    static_assert((GroupsPerWarp % 2) == 0, "Q4 GEMV static ownership preserves scale pairs");

    constexpr int kActiveCodeVectors =
        GroupsPerWarp * Q4RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
    auto* shared_codes       = shared_tiles.codes[cta_warp][0];
    const auto* global_codes = reinterpret_cast<const uint4*>(
        code_row + static_cast<std::int64_t>(group_begin) * Q4RowSplitStorage::kCodeBytesPerGroup);
    if (lane < kActiveCodeVectors) { shared_codes[lane] = global_codes[lane]; }
    __syncwarp();

    std::uint16_t lane_scale_bits = 0;
    if (lane < GroupsPerWarp) {
        lane_scale_bits =
            load_vec<std::uint16_t>(scale_row + static_cast<std::int64_t>(group_begin + lane) *
                                                    Q4RowSplitStorage::kScaleBytesPerGroup);
    }

    const auto* tile_codes       = reinterpret_cast<const std::uint8_t*>(shared_codes);
    const auto* activation_pairs = reinterpret_cast<const __nv_bfloat162*>(activation);
    float accumulator            = 0.0f;
#pragma unroll
    for (int local_group = 0; local_group < GroupsPerWarp; ++local_group) {
        const std::uint16_t scale_bits =
            static_cast<std::uint16_t>(__shfl_sync(kFullWarpMask, lane_scale_bits, local_group));
        const float scale = __half2float(__ushort_as_half(scale_bits));
        const std::uint8_t packed =
            tile_codes[local_group * Q4RowSplitStorage::kCodeBytesPerGroup + lane];
        const int q0      = (static_cast<int>(packed & 0x0fu) ^ 0x08) - 0x08;
        const int q1      = (static_cast<int>(packed >> 4) ^ 0x08) - 0x08;
        const int k_begin = (group_begin + local_group) * Q4RowSplitStorage::kGroupK + lane * 2;
        const float2 activation_pair = __bfloat1622float2(activation_pairs[k_begin >> 1]);
        accumulator = fmaf(static_cast<float>(q0) * scale, activation_pair.x, accumulator);
        accumulator = fmaf(static_cast<float>(q1) * scale, activation_pair.y, accumulator);
    }
    __syncwarp();
    return accumulator;
}

template <class Schedule>
__device__ __forceinline__ float q4_gemv_dot_byte_sync(Q4GemvTileStorage<Schedule>& shared_tiles,
                                                       int cta_warp,
                                                       const __nv_bfloat16* __restrict__ activation,
                                                       const std::uint8_t* __restrict__ code_row,
                                                       const std::uint8_t* __restrict__ scale_row,
                                                       int group_begin, int group_end, int lane) {
    static_assert(Schedule::kLaneMapping == Q4GemvLaneMapping::PackedByte2);
    static_assert(Schedule::kDecodeMode == Q4GemvDecodeMode::ScalarInteger);
    static_assert(Schedule::kCodeTransfer == Q4GemvCodeTransfer::SyncVector16);
    static_assert(Schedule::kScaleAccess == Q4GemvScaleAccess::Scalar16Shuffle);
    static_assert(Schedule::kPipelineStages == 1);

    auto* shared_codes = shared_tiles.codes[cta_warp][0];
    float accumulator0 = 0.0f;
    float accumulator1 = 0.0f;

    for (int tile_group_begin = group_begin; tile_group_begin < group_end;
         tile_group_begin += Schedule::kGroupsPerWarpTile) {
        const int active_groups = min(Schedule::kGroupsPerWarpTile, group_end - tile_group_begin);
        const int active_code_vectors =
            active_groups * Q4RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
        static_assert(Schedule::kCodeVectorsPerTile <= 32,
                      "sync vector transfer currently maps one vector to each lane");
        const auto* global_codes =
            reinterpret_cast<const uint4*>(code_row + static_cast<std::int64_t>(tile_group_begin) *
                                                          Q4RowSplitStorage::kCodeBytesPerGroup);
        if (lane < active_code_vectors) { shared_codes[lane] = global_codes[lane]; }
        __syncwarp();

        std::uint16_t lane_scale_bits = 0;
        if (lane < active_groups) {
            lane_scale_bits = load_vec<std::uint16_t>(
                scale_row + static_cast<std::int64_t>(tile_group_begin + lane) *
                                Q4RowSplitStorage::kScaleBytesPerGroup);
        }

        const auto* tile_codes = reinterpret_cast<const std::uint8_t*>(shared_codes);
#pragma unroll 1
        for (int local_group = 0; local_group < active_groups; local_group += 2) {
            const std::uint16_t scale_bits0 = static_cast<std::uint16_t>(
                __shfl_sync(kFullWarpMask, lane_scale_bits, local_group));
            const std::uint16_t scale_bits1 = static_cast<std::uint16_t>(
                __shfl_sync(kFullWarpMask, lane_scale_bits, local_group + 1));
            const float scale0 = __half2float(__ushort_as_half(scale_bits0));
            const float scale1 = __half2float(__ushort_as_half(scale_bits1));

            const std::uint8_t packed0 =
                tile_codes[local_group * Q4RowSplitStorage::kCodeBytesPerGroup + lane];
            const std::uint8_t packed1 =
                tile_codes[(local_group + 1) * Q4RowSplitStorage::kCodeBytesPerGroup + lane];
            const int q00 = (static_cast<int>(packed0 & 0x0fu) ^ 0x08) - 0x08;
            const int q01 = (static_cast<int>(packed0 >> 4) ^ 0x08) - 0x08;
            const int q10 = (static_cast<int>(packed1 & 0x0fu) ^ 0x08) - 0x08;
            const int q11 = (static_cast<int>(packed1 >> 4) ^ 0x08) - 0x08;

            const int k_begin0 =
                (tile_group_begin + local_group) * Q4RowSplitStorage::kGroupK + lane * 2;
            const int k_begin1           = k_begin0 + Q4RowSplitStorage::kGroupK;
            const auto* activation_pairs = reinterpret_cast<const __nv_bfloat162*>(activation);
            const float2 x0              = __bfloat1622float2(activation_pairs[k_begin0 >> 1]);
            const float2 x1              = __bfloat1622float2(activation_pairs[k_begin1 >> 1]);

            accumulator0 = fmaf(static_cast<float>(q00) * scale0, x0.x, accumulator0);
            accumulator0 = fmaf(static_cast<float>(q01) * scale0, x0.y, accumulator0);
            accumulator1 = fmaf(static_cast<float>(q10) * scale1, x1.x, accumulator1);
            accumulator1 = fmaf(static_cast<float>(q11) * scale1, x1.y, accumulator1);
        }
        __syncwarp();
    }
    return accumulator0 + accumulator1;
}

template <bool SplitOutput, int SplitRow>
__device__ __forceinline__ void q4_gemv_store(__nv_bfloat16* out, __nv_bfloat16* out_tail,
                                              int output_row, float value) {
    if constexpr (SplitOutput) {
        if (output_row < SplitRow) {
            out[output_row] = __float2bfloat16(value);
        } else {
            out_tail[output_row - SplitRow] = __float2bfloat16(value);
        }
    } else {
        out[output_row] = __float2bfloat16(value);
    }
}

// clang-format off
struct Q4GemvStoreEpilogue {
    template <bool SplitOutput, int SplitRow>
    __device__ __forceinline__ void operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail,
                                               int row, float value) const {
        q4_gemv_store<SplitOutput, SplitRow>(out, out_tail, row, value);
    }
};

template <class Schedule, bool SplitOutput = false, int SplitRow = 0,
          class Epilogue = Q4GemvStoreEpilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kLaunchBoundsMinBlocks)
void q4_rowsplit_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ out_tail,
    std::int32_t rows,
    std::int32_t k,
    Epilogue epilogue = {}) {
    // clang-format on
    constexpr int kRowsPerCta  = Schedule::kRowsPerCta;
    constexpr int kWarpsPerRow = Schedule::kWarpsPerRow;
    static_assert(!SplitOutput || SplitRow > 0,
                  "split-output Q4 GEMV requires a positive compile-time seam");

    __shared__ Q4GemvTileStorage<Schedule> shared_tiles;
    __shared__ float row_partials[kRowsPerCta][kWarpsPerRow];
    extern __shared__ __align__(16) unsigned char dynamic_shared[];

    auto* shared_x = reinterpret_cast<__nv_bfloat16*>(dynamic_shared);
    if constexpr (Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK) {
        const auto* global_vectors = reinterpret_cast<const uint4*>(x);
        auto* shared_vectors       = reinterpret_cast<uint4*>(shared_x);
        const int vector_count     = k / 8;
        for (int vector = static_cast<int>(threadIdx.x); vector < vector_count;
             vector += Schedule::kThreads) {
            cp_async<16>(&shared_vectors[vector], &global_vectors[vector]);
        }
        cp_commit();
        cp_wait<0>();
        __syncthreads();
    }

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int cta_warp    = static_cast<int>(threadIdx.x) >> 5;
    const int row_in_cta  = cta_warp / kWarpsPerRow;
    const int warp_in_row = cta_warp % kWarpsPerRow;
    const int row         = static_cast<int>(blockIdx.x) * kRowsPerCta + row_in_cta;

    const int groups_per_row = Schedule::kStaticGroupsPerRow > 0 ? Schedule::kStaticGroupsPerRow
                                                                 : k / Q4RowSplitStorage::kGroupK;
    int group_begin;
    int group_end;
    if constexpr (Schedule::kStaticGroupsPerRow > 0) {
        constexpr int kPairsPerRow = Schedule::kStaticGroupsPerRow / 2;
        static_assert((kPairsPerRow % kWarpsPerRow) == 0,
                      "static Q4 GEMV ownership must divide scale pairs across row warps");
        constexpr int kGroupsPerWarp = Schedule::kStaticGroupsPerRow / kWarpsPerRow;
        group_begin                  = warp_in_row * kGroupsPerWarp;
        group_end                    = group_begin + kGroupsPerWarp;
    } else {
        const int pairs_per_row = groups_per_row / 2;
        const int pair_begin    = pairs_per_row * warp_in_row / kWarpsPerRow;
        const int pair_end      = pairs_per_row * (warp_in_row + 1) / kWarpsPerRow;
        group_begin             = pair_begin * 2;
        group_end               = pair_end * 2;
    }

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * groups_per_row *
                                               Q4RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * groups_per_row *
                                                 Q4RowSplitStorage::kScaleBytesPerGroup;
    const __nv_bfloat16* activation =
        Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK ? shared_x : x;

    float accumulator = 0.0f;
    if constexpr (Schedule::kLaneMapping == Q4GemvLaneMapping::PackedByte2 &&
                  Schedule::kStaticGroupsPerRow > 0) {
        constexpr int kGroupsPerWarp = Schedule::kStaticGroupsPerRow / kWarpsPerRow;
        accumulator                  = q4_gemv_dot_byte_static<Schedule, kGroupsPerWarp>(
            shared_tiles, cta_warp, activation, code_row, scale_row, group_begin, lane);
    } else {
        if (group_begin < group_end) {
            if constexpr (Schedule::kLaneMapping == Q4GemvLaneMapping::PackedByte2) {
                accumulator =
                    q4_gemv_dot_byte_sync<Schedule>(shared_tiles, cta_warp, activation, code_row,
                                                    scale_row, group_begin, group_end, lane);
            } else {
                accumulator =
                    q4_gemv_dot_word_async<Schedule>(shared_tiles, cta_warp, activation, code_row,
                                                     scale_row, group_begin, group_end, lane);
            }
        }
    }

    accumulator = warp_reduce_sum(accumulator);
    if constexpr (kWarpsPerRow == 1) {
        if (lane == 0) {
            epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, row, accumulator);
        }
    } else {
        if (lane == 0) { row_partials[row_in_cta][warp_in_row] = accumulator; }
        __syncthreads();

        if (warp_in_row == 0 && lane == 0) {
            float row_accumulator = 0.0f;
#pragma unroll
            for (int warp = 0; warp < kWarpsPerRow; ++warp) {
                row_accumulator += row_partials[row_in_cta][warp];
            }
            epilogue.template operator()<SplitOutput, SplitRow>(out, out_tail, row,
                                                                row_accumulator);
        }
    }
}

} // namespace ninfer::ops::detail
