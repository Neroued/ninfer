#pragma once

#include "ops/common/warp.cuh"
#include "ops/linear/q6/q6_schedule.cuh"

namespace ninfer::ops::detail {

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

    const std::int64_t group0 = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerWarpStage;
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
                      float (&acc)[Schedule::kBlockTokens]) {
    constexpr int kCols       = Schedule::kBlockTokens;
    constexpr int kCodePhases = Schedule::kCodePhases;

#pragma unroll
    for (int phase = 0; phase < kCodePhases; ++phase) {
        const int group        = phase * 4 + (lane >> 3);
        const int stage_groups = FullStage ? Schedule::kGroupsPerWarpStage : active_groups;
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

// Warps split a row at scale-pair boundaries. With one warp per row the
// original warp-local pipeline is retained and the CTA reduction compiles out.
template <class Schedule, class Output, class Epilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q6_a16_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int tokens, int padded_k, int token_begin) {
    constexpr int R = Schedule::kBlockRows;
    constexpr int W = Schedule::kWarpsPerRow;
    constexpr int T = Schedule::kBlockTokens;
    constexpr int S = Schedule::kStages;
    constexpr int G = Schedule::kGroupsPerWarpStage;

    union SharedStorage {
        struct {
            alignas(16) uint4 codes[Schedule::kWarps][S][Schedule::kCodeVecsPerStage];
            alignas(16) uint4 high[Schedule::kWarps][S][Schedule::kHighVecsPerStage];
            alignas(16) std::uint32_t scales[Schedule::kWarps][S][Schedule::kScalePairsPerStage];
        } staging;

        float partial[R][W][T];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    __shared__ SharedStorage shared;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int local_row = warp / W;
    const int split     = warp % W;
    const int row       = static_cast<int>(blockIdx.x) * R + local_row;
    if constexpr (W == 1) {
        if (row >= rows) return;
    }
    const int safe_row = row < rows ? row : 0;
    int group_begin    = 0;
    int groups         = k / 64;
    if constexpr (W > 1) {
        const int pairs = k / 128;
        group_begin     = (pairs * split / W) * 2;
        groups          = row < rows ? (pairs * (split + 1) / W) * 2 - group_begin : 0;
    }
    const int stages          = (groups + G - 1) / G;
    const int token0          = static_cast<int>(blockIdx.y) * T;
    const int active_tokens   = min(T, tokens - token0);
    const std::int64_t group0 = static_cast<std::int64_t>(safe_row) * (padded_k / 64) + group_begin;
    const auto* code_row      = codes + group0 * 32;
    const auto* high_row      = high + group0 * 16;
    const auto* scale_row     = scales + group0 * 2;
    const auto* warp_x        = x + group_begin * 64;
    auto& stage_codes         = shared.staging.codes;
    auto& stage_high          = shared.staging.high;
    auto& stage_scales        = shared.staging.scales;
    float acc[T]              = {};
#pragma unroll
    for (int prefetch = 0; prefetch < S - 1; ++prefetch) {
        if (prefetch < stages) {
            q6_simt_issue_stage<Schedule, false>(
                stage_codes[warp][prefetch], stage_high[warp][prefetch],
                stage_scales[warp][prefetch], code_row, high_row, scale_row, prefetch,
                min(G, groups - prefetch * G), lane);
        } else {
            cp_commit();
        }
    }
#pragma unroll 1
    for (int stage = 0; stage < stages; ++stage) {
        const int fetch = stage + S - 1;
        if (fetch < stages) {
            q6_simt_issue_stage<Schedule, false>(
                stage_codes[warp][fetch % S], stage_high[warp][fetch % S],
                stage_scales[warp][fetch % S], code_row, high_row, scale_row, fetch,
                min(G, groups - fetch * G), lane);
        } else {
            cp_commit();
        }
        cp_wait<S - 1>();
        __syncwarp();
        q6_simt_consume_stage<Schedule, false, false>(
            warp_x, k, token0, active_tokens, stage, min(G, groups - stage * G),
            stage_codes[warp][stage % S], stage_high[warp][stage % S],
            stage_scales[warp][stage % S], lane, acc);
        __syncwarp();
    }
    if constexpr (W > 1) {
        cp_wait<0>();
        __syncthreads(); // All staged reads and asynchronous writes finish before reuse.
    }
#pragma unroll
    for (int token = 0; token < T; ++token) {
        const float sum = warp_reduce_sum(acc[token]);
        if constexpr (W == 1) {
            if (lane == 0 && token < active_tokens) {
                const int global_token = token_begin + token0 + token;
                output.store(row, global_token, epilogue.apply(row, global_token, sum));
            }
        } else {
            if (lane == 0) shared.partial[local_row][split][token] = sum;
        }
    }
    if constexpr (W > 1) {
        __syncthreads();
        if (split == 0 && lane == 0 && row < rows) {
#pragma unroll
            for (int token = 0; token < T; ++token) {
                if (token < active_tokens) {
                    float sum = shared.partial[local_row][0][token];
#pragma unroll
                    for (int part = 1; part < W; ++part)
                        sum += shared.partial[local_row][part][token];
                    const int global_token = token_begin + token0 + token;
                    output.store(row, global_token, epilogue.apply(row, global_token, sum));
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
