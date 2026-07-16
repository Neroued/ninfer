#pragma once

// Q6G64 RowSplit x BF16 SIMT GEMM.
//
// out[Rows, Cols] = W[Rows, K] * x[K, Cols]
//
// One warp owns one output row across ColsPerTile columns. Raw Q6 code/high
// planes and adjacent FP16 scale pairs are staged per row with cp.async;
// decoded FP32 weights are reused across the column tile and accumulated with
// FP32 FMA.
//
// K is staged in whole quant groups. A predicated final stage still copies
// complete 32-byte code groups and aligned 4-byte pairs of FP16 scales. It
// never falls back to scalar code-pair loads, and lanes belonging to inactive
// groups do not form or read an activation address.

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/q6/q6_rowsplit_launch.h"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int RowsPerCta_, int ColsPerTile_, int GroupsPerStage_, int PipelineStages_,
          Cache CodeCache_, int LaunchBoundsMinBlocks_>
struct Q6RowSplitSimtGemmSchedule {
    static constexpr int kRowsPerCta            = RowsPerCta_;
    static constexpr int kColsPerTile           = ColsPerTile_;
    static constexpr int kGroupsPerStage        = GroupsPerStage_;
    static constexpr int kPipelineStages        = PipelineStages_;
    static constexpr Cache kCodeCache           = CodeCache_;
    static constexpr int kLaunchBoundsMinBlocks = LaunchBoundsMinBlocks_;

    static constexpr int kCtaWarps = kRowsPerCta;
    static constexpr int kThreads  = kCtaWarps * 32;
    static constexpr int kStageK   = kGroupsPerStage * Q6RowSplitStorage::kGroupK;
    static constexpr int kCodeVecsPerStage =
        kGroupsPerStage * Q6RowSplitStorage::kCodeBytesPerGroup / static_cast<int>(sizeof(uint4));
    static constexpr int kHighVecsPerStage =
        kGroupsPerStage * Q6RowSplitStorage::kHighBytesPerGroup / static_cast<int>(sizeof(uint4));
    static constexpr int kScalePairsPerStage = kGroupsPerStage / 2;
    static constexpr int kCodePhases         = (kGroupsPerStage + 3) / 4;
    static constexpr int kSharedBytes =
        kRowsPerCta * kPipelineStages *
        (kGroupsPerStage * Q6RowSplitStorage::kCodeBytesPerGroup +
         kGroupsPerStage * Q6RowSplitStorage::kHighBytesPerGroup +
         kScalePairsPerStage * static_cast<int>(sizeof(std::uint32_t)));

    static_assert(kRowsPerCta > 0 && kRowsPerCta <= 32);
    static_assert(kColsPerTile > 0 && kColsPerTile <= 8);
    static_assert(kGroupsPerStage > 0 && kGroupsPerStage % 2 == 0,
                  "Q6 SIMT stages load aligned pairs of FP16 scales");
    static_assert(kPipelineStages >= 2 && kPipelineStages <= 8,
                  "Q6 SIMT cp.async pipeline depth must fit cp_wait");
    static_assert(kLaunchBoundsMinBlocks >= 1);
    static_assert(kThreads <= 1024);
    static_assert(kCodeVecsPerStage * static_cast<int>(sizeof(uint4)) ==
                      kGroupsPerStage * Q6RowSplitStorage::kCodeBytesPerGroup,
                  "Q6 code groups must decompose into complete 16-byte vectors");
    static_assert(kHighVecsPerStage * static_cast<int>(sizeof(uint4)) ==
                      kGroupsPerStage * Q6RowSplitStorage::kHighBytesPerGroup,
                  "Q6 high groups must decompose into complete 16-byte vectors");
    static_assert(kSharedBytes <= 48 * 1024,
                  "Q6 SIMT staged shared memory exceeds the static 48 KiB budget");
};

template <class Schedule>
__device__ __forceinline__ void q6_simt_copy_code(uint4* shared_dst,
                                                  const std::uint8_t* global_src) {
    if constexpr (Schedule::kCodeCache == Cache::cg) {
        cp_async<16, Cache::cg>(shared_dst, global_src);
    } else {
        cp_async<16, Cache::ca>(shared_dst, global_src);
    }
}

template <class Schedule, bool FullStage>
__device__ __forceinline__ void q6_simt_issue_stage(uint4* __restrict__ shared_codes,
                                                    uint4* __restrict__ shared_high,
                                                    std::uint32_t* __restrict__ shared_scales,
                                                    const std::uint8_t* __restrict__ code_row,
                                                    const std::uint8_t* __restrict__ high_row,
                                                    const std::uint8_t* __restrict__ scale_row,
                                                    int stage, int active_groups, int lane) {
    constexpr int kCodeVecs   = Schedule::kCodeVecsPerStage;
    constexpr int kHighVecs   = Schedule::kHighVecsPerStage;
    constexpr int kScalePairs = Schedule::kScalePairsPerStage;

    const std::int64_t group0        = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerStage;
    const std::uint8_t* stage_codes  = code_row + group0 * Q6RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* stage_high   = high_row + group0 * Q6RowSplitStorage::kHighBytesPerGroup;
    const std::uint8_t* stage_scales = scale_row + group0 * Q6RowSplitStorage::kScaleBytesPerGroup;

    const int active_code_vecs = FullStage ? kCodeVecs
                                           : active_groups * Q6RowSplitStorage::kCodeBytesPerGroup /
                                                 static_cast<int>(sizeof(uint4));
    for (int vec = lane; vec < kCodeVecs; vec += 32) {
        if (FullStage || vec < active_code_vecs) {
            q6_simt_copy_code<Schedule>(&shared_codes[vec],
                                        stage_codes + static_cast<std::int64_t>(vec) * 16);
        } else {
            shared_codes[vec] = uint4{0u, 0u, 0u, 0u};
        }
    }

    const int active_high_vecs = FullStage ? kHighVecs
                                           : active_groups * Q6RowSplitStorage::kHighBytesPerGroup /
                                                 static_cast<int>(sizeof(uint4));
    for (int vec = lane; vec < kHighVecs; vec += 32) {
        if (FullStage || vec < active_high_vecs) {
            q6_simt_copy_code<Schedule>(&shared_high[vec],
                                        stage_high + static_cast<std::int64_t>(vec) * 16);
        } else {
            shared_high[vec] = uint4{0u, 0u, 0u, 0u};
        }
    }

    const int active_scale_pairs = FullStage ? kScalePairs : active_groups / 2;
    for (int pair = lane; pair < kScalePairs; pair += 32) {
        if (FullStage || pair < active_scale_pairs) {
            cp_async<4>(&shared_scales[pair], stage_scales + static_cast<std::int64_t>(pair) * 4);
        } else {
            shared_scales[pair] = 0u;
        }
    }
    cp_commit();
}

template <class Schedule, bool FullStage, bool FullCols>
__device__ __forceinline__ void
q6_simt_consume_stage(const __nv_bfloat16* __restrict__ x, std::int32_t k, int col0,
                      int active_cols, int stage, int active_groups,
                      const uint4* __restrict__ shared_codes, const uint4* __restrict__ shared_high,
                      const std::uint32_t* __restrict__ shared_scales, int lane,
                      float (&acc)[Schedule::kColsPerTile]) {
    constexpr int kCols       = Schedule::kColsPerTile;
    constexpr int kCodePhases = Schedule::kCodePhases;

#pragma unroll
    for (int phase = 0; phase < kCodePhases; ++phase) {
        const int group        = phase * 4 + (lane >> 3);
        const int stage_groups = FullStage ? Schedule::kGroupsPerStage : active_groups;
        if (group < stage_groups) {
            const std::uint32_t packed =
                reinterpret_cast<const std::uint32_t*>(shared_codes)[phase * 32 + lane];
            const std::uint16_t high_bits =
                reinterpret_cast<const std::uint16_t*>(shared_high)[phase * 32 + lane];
            const std::uint32_t scale_pair = shared_scales[group >> 1];
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(scale_pair >> ((group & 1) * 16));

            float weights[8];
            Q6SimtDecodeAtom::decode_eight(packed, high_bits, scale_bits, weights);

            const std::int64_t xk = static_cast<std::int64_t>(stage) * Schedule::kStageK +
                                    static_cast<std::int64_t>(phase) * 256 + lane * 8;
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                if (FullCols || col < active_cols) {
                    const uint4 values =
                        load_vec<uint4>(x + static_cast<std::int64_t>(col0 + col) * k + xk);
                    const float2 x0 = bf16x2_bits_to_float2(values.x);
                    const float2 x1 = bf16x2_bits_to_float2(values.y);
                    const float2 x2 = bf16x2_bits_to_float2(values.z);
                    const float2 x3 = bf16x2_bits_to_float2(values.w);
                    acc[col]        = fmaf(weights[0], x0.x, acc[col]);
                    acc[col]        = fmaf(weights[1], x0.y, acc[col]);
                    acc[col]        = fmaf(weights[2], x1.x, acc[col]);
                    acc[col]        = fmaf(weights[3], x1.y, acc[col]);
                    acc[col]        = fmaf(weights[4], x2.x, acc[col]);
                    acc[col]        = fmaf(weights[5], x2.y, acc[col]);
                    acc[col]        = fmaf(weights[6], x3.x, acc[col]);
                    acc[col]        = fmaf(weights[7], x3.y, acc[col]);
                }
            }
        }
    }
}

template <class Schedule, Q6KernelVariant Variant>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::
        kLaunchBoundsMinBlocks) void q6_rowsplit_gemm_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                                                  const std::
                                                                      uint8_t* __restrict__ codes,
                                                                  const std::
                                                                      uint8_t* __restrict__ high,
                                                                  const std::
                                                                      uint8_t* __restrict__ scales,
                                                                  __nv_bfloat16* __restrict__ out,
                                                                  std::int32_t rows, std::int32_t k,
                                                                  std::int32_t cols,
                                                                  std::int32_t padded_k) {
    static_assert(Variant == Q6KernelVariant::Full || Variant == Q6KernelVariant::Predicated,
                  "Q6 SIMT GEMM requires a tiled kernel variant");

    constexpr bool kFull              = Variant == Q6KernelVariant::Full;
    constexpr int kRowsPerCta         = Schedule::kRowsPerCta;
    constexpr int kColsPerTile        = Schedule::kColsPerTile;
    constexpr int kGroupsPerStage     = Schedule::kGroupsPerStage;
    constexpr int kPipelineStages     = Schedule::kPipelineStages;
    constexpr int kPipelinePrefetch   = kPipelineStages - 1;
    constexpr int kCodeVecsPerStage   = Schedule::kCodeVecsPerStage;
    constexpr int kHighVecsPerStage   = Schedule::kHighVecsPerStage;
    constexpr int kScalePairsPerStage = Schedule::kScalePairsPerStage;

    __shared__ __align__(16) uint4 shared_codes[kRowsPerCta][kPipelineStages][kCodeVecsPerStage];
    __shared__ __align__(16) uint4 shared_high[kRowsPerCta][kPipelineStages][kHighVecsPerStage];
    __shared__ __align__(16)
        std::uint32_t shared_scales[kRowsPerCta][kPipelineStages][kScalePairsPerStage];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerCta + warp;
    if constexpr (!kFull) {
        if (row >= rows) { return; }
    }

    const int col0        = static_cast<int>(blockIdx.y) * kColsPerTile;
    const int active_cols = kFull ? kColsPerTile : min(kColsPerTile, cols - col0);

    const int padded_groups = padded_k / Q6RowSplitStorage::kGroupK;
    const int groups        = k / Q6RowSplitStorage::kGroupK;
    const int stages =
        kFull ? groups / kGroupsPerStage : (groups + kGroupsPerStage - 1) / kGroupsPerStage;

    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(row) * padded_groups *
                                               Q6RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* high_row = high + static_cast<std::int64_t>(row) * padded_groups *
                                              Q6RowSplitStorage::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * padded_groups *
                                                 Q6RowSplitStorage::kScaleBytesPerGroup;

    float acc[kColsPerTile];
#pragma unroll
    for (int col = 0; col < kColsPerTile; ++col) { acc[col] = 0.0f; }

#pragma unroll
    for (int prefetch = 0; prefetch < kPipelinePrefetch; ++prefetch) {
        if (prefetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - prefetch * kGroupsPerStage);
            q6_simt_issue_stage<Schedule, kFull>(shared_codes[warp][prefetch],
                                                 shared_high[warp][prefetch],
                                                 shared_scales[warp][prefetch], code_row, high_row,
                                                 scale_row, prefetch, active_groups, lane);
        } else {
            cp_commit();
        }
    }

#pragma unroll 1
    for (int stage = 0; stage < stages; ++stage) {
        const int fetch = stage + kPipelinePrefetch;
        if (fetch < stages) {
            const int active_groups =
                kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - fetch * kGroupsPerStage);
            const int buffer = fetch % kPipelineStages;
            q6_simt_issue_stage<Schedule, kFull>(
                shared_codes[warp][buffer], shared_high[warp][buffer], shared_scales[warp][buffer],
                code_row, high_row, scale_row, fetch, active_groups, lane);
        } else {
            cp_commit();
        }

        cp_wait<kPipelinePrefetch>();
        __syncwarp();

        const int active_groups =
            kFull ? kGroupsPerStage : min(kGroupsPerStage, groups - stage * kGroupsPerStage);
        const int buffer = stage % kPipelineStages;
        q6_simt_consume_stage<Schedule, kFull, kFull>(
            x, k, col0, active_cols, stage, active_groups, shared_codes[warp][buffer],
            shared_high[warp][buffer], shared_scales[warp][buffer], lane, acc);
        __syncwarp();
    }

#pragma unroll
    for (int col = 0; col < kColsPerTile; ++col) {
        if (kFull || col < active_cols) {
            const float sum = warp_reduce_sum(acc[col]);
            if (lane == 0) {
                out[static_cast<std::int64_t>(col0 + col) * rows + row] = __float2bfloat16(sum);
            }
        }
    }
}

} // namespace ninfer::ops::detail
