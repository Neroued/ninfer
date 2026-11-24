#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/q8/q8_schedule.cuh"
#include "ops/linear/kernels/q8/q8_operands.h"

namespace ninfer::ops::detail {
template <class Schedule, bool FullRows, class Output, class Epilogue>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q8_a16_gemv_kernel(
    Q8LinearOperands operands, Output output, Epilogue epilogue) {
    constexpr int R = Schedule::kBlockRows, W = Schedule::kWarpsPerRow;
    const int k        = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.k;
    const int padded_k = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.padded_k;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int local_row = warp / W, split = warp % W;
    const int row = blockIdx.x * R + local_row;
    float acc     = 0.0f;
    if (FullRows || row < operands.rows) {
        const auto* code_row = operands.codes + static_cast<std::int64_t>(row) * padded_k;
        const auto* scale_row =
            operands.scales + static_cast<std::int64_t>(row) * (padded_k / 32) * 2;
#pragma unroll Schedule::kUnroll
        for (int phase = split; phase < (k + 255) / 256; phase += W) {
            unsigned bits = 0;
            if (lane < 8 && (Schedule::kStaticK > 0 || phase * 256 + lane * 32 < k))
                bits = common::load_ldg<std::uint16_t>(scale_row + (phase * 8 + lane) * 2);
            bits              = __shfl_sync(0xffffffffu, bits, lane >> 2);
            const float scale = __half2float(__ushort_as_half(bits));
            const int kk      = phase * 256 + lane * 8;
            if (Schedule::kStaticK > 0 || kk < k) {
                const uint2 codes  = common::load_ldg<uint2>(code_row + kk);
                const uint4 values = common::load_ldg<uint4>(operands.x + kk);
                const float2 xv[]{common::bf16x2_bits_to_float2(values.x),
                                  common::bf16x2_bits_to_float2(values.y),
                                  common::bf16x2_bits_to_float2(values.z),
                                  common::bf16x2_bits_to_float2(values.w)};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const unsigned word = i < 4 ? codes.x : codes.y;
                    const float weight =
                        static_cast<float>(static_cast<std::int8_t>(word >> ((i & 3) * 8))) * scale;
                    acc = fmaf(weight, (i & 1) ? xv[i / 2].y : xv[i / 2].x, acc);
                }
            }
        }
    }
    acc = common::warp_reduce_sum(acc);
    if constexpr (W > 1) {
        __shared__ float partial[R][W];
        if (lane == 0) partial[local_row][split] = acc;
        __syncthreads();
        if (lane == 0 && split == 0 && (FullRows || row < operands.rows)) {
            float values[1]{partial[local_row][0]};
#pragma unroll
            for (int i = 1; i < W; ++i) values[0] += partial[local_row][i];
            linear_finish_row(linear_output_tile<R>(output, blockIdx.x * R), epilogue, row, 0,
                              values, 1);
        }
    } else if (lane == 0 && (FullRows || row < operands.rows)) {
        const float values[1]{acc};
        linear_finish_row(linear_output_tile<R>(output, blockIdx.x * R), epilogue, row, 0, values,
                          1);
    }
}

// Reuse each decoded FP32 weight across a bounded group of input vectors.
// The CTA owns complete output rows; only its K partitions need shared reduction.
template <class Schedule, bool FullRows, class Output, class Epilogue>
__global__
__launch_bounds__(Schedule::kThreads,
                  Schedule::kMinBlocksPerSm) void q8_a16_gemv_tile_kernel(Q8LinearOperands operands,
                                                                          Output output,
                                                                          Epilogue epilogue) {
    constexpr int R = Schedule::kBlockRows, W = Schedule::kWarpsPerRow;
    constexpr int T    = Schedule::kBlockTokens;
    const int k        = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.k;
    const int padded_k = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.padded_k;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int local_row = warp / W, split = warp % W;
    const int row = blockIdx.x * R + local_row;
    float acc[T]{};
    if (FullRows || row < operands.rows) {
        const auto* code_row  = operands.codes + std::int64_t(row) * padded_k;
        const auto* scale_row = operands.scales + std::int64_t(row) * (padded_k / 32) * 2;
#pragma unroll Schedule::kUnroll
        for (int phase = split; phase < (k + 255) / 256; phase += W) {
            unsigned bits = 0;
            if (lane < 8 && (Schedule::kStaticK > 0 || phase * 256 + lane * 32 < k))
                bits = common::load_ldg<std::uint16_t>(scale_row + (phase * 8 + lane) * 2);
            bits              = __shfl_sync(0xffffffffu, bits, lane >> 2);
            const float scale = __half2float(__ushort_as_half(bits));
            const int kk      = phase * 256 + lane * 8;
            if (Schedule::kStaticK > 0 || kk < k) {
                const uint2 packed = common::load_ldg<uint2>(code_row + kk);
                float weights[8];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    weights[i] = float(static_cast<std::int8_t>((i < 4 ? packed.x : packed.y) >>
                                                                ((i & 3) * 8))) *
                                 scale;
#pragma unroll
                for (int token = 0; token < T; ++token) {
                    if (token < operands.tokens) {
                        const uint4 values =
                            common::load_ldg<uint4>(operands.x + std::int64_t(token) * k + kk);
                        const float2 xv[]{common::bf16x2_bits_to_float2(values.x),
                                          common::bf16x2_bits_to_float2(values.y),
                                          common::bf16x2_bits_to_float2(values.z),
                                          common::bf16x2_bits_to_float2(values.w)};
#pragma unroll
                        for (int i = 0; i < 8; ++i)
                            acc[token] =
                                fmaf(weights[i], (i & 1) ? xv[i / 2].y : xv[i / 2].x, acc[token]);
                    }
                }
            }
        }
    }
    __shared__ float partial[R][W][T];
#pragma unroll
    for (int token = 0; token < T; ++token) {
        acc[token] = common::warp_reduce_sum(acc[token]);
        if (lane == 0) partial[local_row][split][token] = acc[token];
    }
    __syncthreads();
    if (lane == 0 && split == 0 && (FullRows || row < operands.rows)) {
        float values[T]{};
#pragma unroll
        for (int token = 0; token < T; ++token) {
#pragma unroll
            for (int part = 0; part < W; ++part) values[token] += partial[local_row][part][token];
        }
        linear_finish_row(linear_output_tile<R>(output, blockIdx.x * R), epilogue, row, 0, values,
                          operands.tokens);
    }
}
} // namespace ninfer::ops::detail
