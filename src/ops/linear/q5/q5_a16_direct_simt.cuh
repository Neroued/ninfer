#pragma once

#include "core/pdl.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q5/q5_schedule.cuh"

namespace ninfer::ops::detail {

// Direct global loads retain the split-2/split-4 SIMT dataflow. Each warp owns
// contiguous 256-value phases; the CTA reduces K before invoking the epilogue.
template <class Schedule, bool FullK, bool FullTokens, class Output, class Epilogue,
          bool TriggerPdl = false, bool JoinPdl = false>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q5_a16_direct_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int tokens, int padded_k, int token_begin) {
    constexpr int R  = Schedule::kBlockRows;
    constexpr int W  = Schedule::kWarpsPerRow;
    constexpr int T  = Schedule::kBlockTokens;
    constexpr int P  = Schedule::kPhasesPerWarp;
    constexpr int BK = Schedule::kBlockK;
    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) pdl::trigger_dependents();
    }
    __shared__ float partial[R][W][T];
    const int warp          = int(threadIdx.x) >> 5;
    const int lane          = int(threadIdx.x) & 31;
    const int local_row     = warp / W;
    const int part          = warp % W;
    const int row           = int(blockIdx.x) * R + local_row;
    const int token0        = Schedule::kExactTokens ? 0 : int(blockIdx.y) * T;
    const int global_token0 = Schedule::kExactTokens ? 0 : token_begin + token0;
    if constexpr (R == 1) {
        if (row >= rows) return;
    }
    const int active_tokens = FullTokens ? T : min(T, tokens - token0);
    const int logical_k     = Schedule::kStaticK ? Schedule::kStaticK : k;
    const int iterations    = (logical_k + BK - 1) / BK;
    const auto group0       = std::int64_t(R == 1 || row < rows ? row : 0) * (padded_k / 64);
    float acc[T]            = {};
    constexpr int kUnroll   = Schedule::kStaticK ? (Schedule::kStaticK + BK - 1) / BK : 1;
#pragma unroll kUnroll
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (int phase = 0; phase < P; ++phase) {
            const int kk             = iteration * BK + (part * P + phase) * 256 + lane * 8;
            const bool active        = FullK || kk + 8 <= logical_k;
            const auto group_index   = group0 + kk / 64;
            std::uint32_t scale_bits = 0;
            if ((lane & 7) == 0 && active)
                scale_bits = load_vec<std::uint16_t>(scales + group_index * 2);
            scale_bits = __shfl_sync(kFullWarpMask, scale_bits, lane & ~7);
            if (active) {
                const auto word = load_vec<std::uint32_t>(codes + group_index * 32 + (kk % 64) / 2);
                const auto high_byte = high[group_index * 8 + (kk % 64) / 8];
                float weights[8];
                Q5SimtDecodeAtom::decode_eight(word, high_byte, scale_bits, weights);
#pragma unroll
                for (int token = 0; token < T; ++token) {
                    if (FullTokens || token < active_tokens) {
                        const uint4 values =
                            load_vec<uint4>(x + std::int64_t(token0 + token) * logical_k + kk);
                        const float2 x0 = bf16x2_bits_to_float2(values.x);
                        const float2 x1 = bf16x2_bits_to_float2(values.y);
                        const float2 x2 = bf16x2_bits_to_float2(values.z);
                        const float2 x3 = bf16x2_bits_to_float2(values.w);
                        acc[token]      = fmaf(weights[0], x0.x, acc[token]);
                        acc[token]      = fmaf(weights[1], x0.y, acc[token]);
                        acc[token]      = fmaf(weights[2], x1.x, acc[token]);
                        acc[token]      = fmaf(weights[3], x1.y, acc[token]);
                        acc[token]      = fmaf(weights[4], x2.x, acc[token]);
                        acc[token]      = fmaf(weights[5], x2.y, acc[token]);
                        acc[token]      = fmaf(weights[6], x3.x, acc[token]);
                        acc[token]      = fmaf(weights[7], x3.y, acc[token]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int token = 0; token < T; ++token) {
        const float sum = warp_reduce_sum(acc[token]);
        if (lane == 0) partial[local_row][part][token] = sum;
    }
    __syncthreads();
    constexpr bool kRowConsumer = requires {
        epilogue.apply_row(output, row, global_token0, partial[local_row][0], active_tokens);
    };
    if (part == 0 && lane < active_tokens) {
        float sum = partial[local_row][0][lane];
#pragma unroll
        for (int split = 1; split < W; ++split) sum += partial[local_row][split][lane];
        if constexpr (kRowConsumer)
            partial[local_row][0][lane] = sum;
        else if (row < rows)
            output.store(row, global_token0 + lane, epilogue.apply(row, global_token0 + lane, sum));
    }
    if constexpr (kRowConsumer) {
        __syncthreads();
        if (part == 0 && lane == 0 && row < rows)
            linear_finish_row(output, epilogue, row, global_token0, partial[local_row][0],
                              active_tokens);
    }
    if constexpr (JoinPdl) pdl::wait_for_dependencies();
}

} // namespace ninfer::ops::detail
