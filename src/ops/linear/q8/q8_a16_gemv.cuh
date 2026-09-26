#pragma once
#include "ops/common/warp.cuh"
#include "ops/common/math.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_operands.h"

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
                bits = load_ldg<std::uint16_t>(scale_row + (phase * 8 + lane) * 2);
            bits              = __shfl_sync(0xffffffffu, bits, lane >> 2);
            const float scale = __half2float(__ushort_as_half(bits));
            const int kk      = phase * 256 + lane * 8;
            if (Schedule::kStaticK > 0 || kk < k) {
                const uint2 codes  = load_ldg<uint2>(code_row + kk);
                const uint4 values = load_ldg<uint4>(operands.x + kk);
                const float2 xv[]{bf16x2_bits_to_float2(values.x), bf16x2_bits_to_float2(values.y),
                                  bf16x2_bits_to_float2(values.z), bf16x2_bits_to_float2(values.w)};
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
    acc = warp_reduce_sum(acc);
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
} // namespace ninfer::ops::detail
