#pragma once

#include "core/pdl.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q5/q5_schedule.cuh"

namespace ninfer::ops::detail {

template <class Schedule>
__device__ __forceinline__ void q5_simt_copy_code(uint4* shared_dst,
                                                  const std::uint8_t* global_src) {
    if constexpr (Schedule::kCodeCache == Cache::cg) {
        cp_async<16, Cache::cg>(shared_dst, global_src);
    } else {
        cp_async<16, Cache::ca>(shared_dst, global_src);
    }
}

template <class Schedule, bool FullStage>
__device__ __forceinline__ void
q5_simt_issue_stage(uint4* __restrict__ shared_codes, std::uint32_t* __restrict__ shared_scales,
                    uint4* __restrict__ shared_high, const std::uint8_t* __restrict__ code_row,
                    const std::uint8_t* __restrict__ scale_row,
                    const std::uint8_t* __restrict__ high_row, int stage, int active_groups,
                    int lane) {
    constexpr int kCodeVecs   = Schedule::kCodeVecsPerStage;
    constexpr int kScalePairs = Schedule::kScalePairsPerStage;

    const std::int64_t group0 = static_cast<std::int64_t>(stage) * Schedule::kGroupsPerWarpStage;
    const std::uint8_t* stage_codes  = code_row + group0 * Q5RowSplitStorage::kCodeBytesPerGroup;
    const std::uint8_t* stage_scales = scale_row + group0 * Q5RowSplitStorage::kScaleBytesPerGroup;

    const int active_code_vecs = FullStage ? kCodeVecs
                                           : active_groups * Q5RowSplitStorage::kCodeBytesPerGroup /
                                                 static_cast<int>(sizeof(uint4));
    for (int vec = lane; vec < kCodeVecs; vec += 32) {
        if (FullStage || vec < active_code_vecs) {
            q5_simt_copy_code<Schedule>(&shared_codes[vec],
                                        stage_codes + static_cast<std::int64_t>(vec) * 16);
        } else {
            shared_codes[vec] = uint4{0u, 0u, 0u, 0u};
        }
    }

    for (int pair = lane; pair < Schedule::kHighVecsPerStage; pair += 32) {
        if (FullStage || pair < active_groups / 2)
            cp_async<16, Schedule::kCodeCache>(&shared_high[pair],
                                               high_row + group0 * 8 + pair * 16);
        else
            shared_high[pair] = uint4{0, 0, 0, 0};
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
__device__ __forceinline__ void q5_simt_consume_stage(
    const __nv_bfloat16* __restrict__ x, std::int32_t k, int col0, int active_cols, int stage,
    int active_groups, const uint4* __restrict__ shared_codes,
    const std::uint32_t* __restrict__ shared_scales, const uint4* __restrict__ shared_high,
    int lane, float (&acc)[Schedule::kBlockTokens]) {
    constexpr int kCols       = Schedule::kBlockTokens;
    constexpr int kCodePhases = Schedule::kCodePhases;

#pragma unroll
    for (int phase = 0; phase < kCodePhases; ++phase) {
        const int group        = phase * 4 + (lane >> 3);
        const int stage_groups = FullStage ? Schedule::kGroupsPerWarpStage : active_groups;
        if (group < stage_groups) {
            const std::uint32_t packed =
                reinterpret_cast<const std::uint32_t*>(shared_codes)[phase * 32 + lane];
            const std::uint32_t scale_pair = shared_scales[group >> 1];
            const std::uint16_t scale_bits =
                static_cast<std::uint16_t>(scale_pair >> ((group & 1) * 16));

            float weights[8];
            Q5SimtDecodeAtom::decode_eight(
                packed, reinterpret_cast<const std::uint8_t*>(shared_high)[phase * 32 + lane],
                scale_bits, weights);

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
template <class Schedule, bool Full, bool FullK, class Output, class Epilogue,
          bool TriggerPdl = false, bool JoinPdl = false>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q5_a16_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int tokens, int padded_k, int token_begin) {
    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) pdl::trigger_dependents();
    }
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
    if constexpr (W == 1 && !Full) {
        if (row >= rows) return;
    }
    const int safe_row      = row < rows ? row : 0;
    int group_begin         = 0;
    const int vector_groups = k % 8 == 0 ? (k / 128) * 2 : 0;
    int groups              = vector_groups;
    if constexpr (W > 1) {
        const int pairs = vector_groups / 2;
        group_begin     = (pairs * split / W) * 2;
        groups          = row < rows ? (pairs * (split + 1) / W) * 2 - group_begin : 0;
    }
    const int stages          = FullK ? groups / G : (groups + G - 1) / G;
    const int token0          = static_cast<int>(blockIdx.y) * T;
    const int active_tokens   = Full ? T : min(T, tokens - token0);
    const std::int64_t group0 = static_cast<std::int64_t>(safe_row) * (padded_k / 64) + group_begin;
    const auto* code_row      = codes + group0 * 32;
    const auto* high_row      = high + group0 * 8;
    const auto* scale_row     = scales + group0 * 2;
    const auto* warp_x        = x + group_begin * 64;
    auto& stage_codes         = shared.staging.codes;
    auto& stage_scales        = shared.staging.scales;
    float acc[T]              = {};
#pragma unroll
    for (int prefetch = 0; prefetch < S - 1; ++prefetch) {
        if (prefetch < stages) {
            q5_simt_issue_stage<Schedule, FullK>(
                stage_codes[warp][prefetch], stage_scales[warp][prefetch],
                shared.staging.high[warp][prefetch], code_row, scale_row, high_row, prefetch,
                min(G, groups - prefetch * G), lane);
        } else {
            cp_commit();
        }
    }
#pragma unroll 1
    for (int stage = 0; stage < stages; ++stage) {
        const int fetch = stage + S - 1;
        if (fetch < stages) {
            q5_simt_issue_stage<Schedule, FullK>(
                stage_codes[warp][fetch % S], stage_scales[warp][fetch % S],
                shared.staging.high[warp][fetch % S], code_row, scale_row, high_row, fetch,
                min(G, groups - fetch * G), lane);
        } else {
            cp_commit();
        }
        cp_wait<S - 1>();
        __syncwarp();
        q5_simt_consume_stage<Schedule, FullK, Full>(
            warp_x, k, token0, active_tokens, stage, min(G, groups - stage * G),
            stage_codes[warp][stage % S], stage_scales[warp][stage % S],
            shared.staging.high[warp][stage % S], lane, acc);
        __syncwarp();
    }
    // Warp zero owns the remainder; other K warps only own complete scale pairs.
    if (!FullK && split == 0 && row < rows) {
        for (int group = vector_groups; group < (k + 63) / 64; ++group) {
            const int kk = group * 64 + lane * 2;
            if (kk < k) {
                float w0, w1;
                Q5ScalarDecodeAtom::load_pair(
                    codes, high, scales, std::int64_t(row) * (padded_k / 64) + group, lane, w0, w1);
#pragma unroll
                for (int token = 0; token < T; ++token) {
                    if (Full || token < active_tokens) {
                        const auto index = std::int64_t(token0 + token) * k + kk;
                        acc[token]       = fmaf(w0, __bfloat162float(x[index]), acc[token]);
                        if (kk + 1 < k)
                            acc[token] = fmaf(w1, __bfloat162float(x[index + 1]), acc[token]);
                    }
                }
            }
        }
    }
    if constexpr (W > 1) {
        cp_wait<0>();
        __syncthreads(); // All staged reads and asynchronous writes finish before reuse.
    }
    float sums[T];
#pragma unroll
    for (int token = 0; token < T; ++token) {
        sums[token] = warp_reduce_sum(acc[token]);
        if constexpr (W > 1) {
            if (lane == 0) shared.partial[local_row][split][token] = sums[token];
        }
    }
    if constexpr (W > 1) {
        __syncthreads();
        if (split == 0 && lane == 0 && row < rows) {
#pragma unroll
            for (int token = 0; token < T; ++token) {
                sums[token] = shared.partial[local_row][0][token];
#pragma unroll
                for (int part = 1; part < W; ++part)
                    sums[token] += shared.partial[local_row][part][token];
            }
            linear_finish_row(output, epilogue, row, token_begin + token0, sums, active_tokens);
        }
    } else if (lane == 0) {
        linear_finish_row(output, epilogue, row, token_begin + token0, sums, active_tokens);
    }
    if constexpr (JoinPdl) { pdl::wait_for_dependencies(); }
}

} // namespace ninfer::ops::detail
