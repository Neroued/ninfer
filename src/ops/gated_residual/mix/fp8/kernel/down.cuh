#pragma once
#include "ops/gated_residual/mix/fp8/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/math.cuh"
#include "ops/common/quant/fp8.cuh"

namespace ninfer::ops::detail::hc::mix::fp8 {
// Partition the contraction so even one column fills the GPU. Injection consumes
// the original BF16 input and shares the same reduction without activation encoding.
template <int Tokens, int Rows, int Values>
__global__ __launch_bounds__(128) void down_split_vector(const __nv_bfloat16* x,
                                                         const std::uint8_t* codes,
                                                         const __nv_bfloat16* injection,
                                                         float* partial, int m, int parts) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int r0   = (blockIdx.x * 4 + warp) * Rows;
    const int part = blockIdx.z, t0 = blockIdx.y * Tokens;
    constexpr int Step = 32 * Values, Chunks = kTotal / Step;
    const int first = part * Chunks / parts, last = (part + 1) * Chunks / parts;
    float sum[Rows][Tokens]{};
    const int nr = injection ? kRank + 4 : kRank;
    for (int chunk = first; chunk < last; ++chunk) {
        const int k = chunk * Step + lane * Values;
        float value[Tokens][Values];
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
#pragma unroll
            for (int v = 0; v < Values / 2; ++v) {
                const auto z        = t0 + t < m
                                          ? __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
                                         x + std::int64_t(t0 + t) * kTotal + k)[v])
                                          : make_float2(0, 0);
                value[t][2 * v]     = z.x;
                value[t][2 * v + 1] = z.y;
            }
        }
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const int r = r0 + row;
            if (r >= nr) continue;
#pragma unroll
            for (int v = 0; v < Values / 2; ++v) {
                float2 w;
                if (r < kRank) {
                    const auto bits = *reinterpret_cast<const unsigned short*>(
                        codes + std::int64_t(r) * kTotal + k + 2 * v);
                    w = common::fp8_code2_to_float2(bits);
                } else
                    w = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
                        injection + (r - kRank) * kTotal + k)[v]);
#pragma unroll
                for (int t = 0; t < Tokens; ++t) {
                    sum[row][t] = fmaf(w.x, value[t][2 * v], sum[row][t]);
                    sum[row][t] = fmaf(w.y, value[t][2 * v + 1], sum[row][t]);
                }
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Rows; ++row)
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            float v = common::warp_sum(sum[row][t]);
            if (lane == 0 && r0 + row < nr && t0 + t < m)
                partial[(std::int64_t(part) * m + t0 + t) * (kRank + 4) + r0 + row] = v;
        }
}

__global__ void down_split_finish(const float* partial, const __nv_bfloat16* scales,
                                  __nv_bfloat16* u, float* injection, int m, int parts) {
    const int row = blockIdx.x * 128 + threadIdx.x, t = blockIdx.y;
    if (row >= kRank + (injection ? 4 : 0)) return;
    float sum = 0;
    for (int part = 0; part < parts; ++part)
        sum += partial[(std::int64_t(part) * m + t) * (kRank + 4) + row];
    if (row < kRank)
        u[t * kRank + row] =
            __float2bfloat16_rn(common::silu(sum * __bfloat162float(scales[row]) * .25F));
    else
        injection[t * 4 + row - kRank] = 2.F * common::sigmoid(sum * .25F);
}

} // namespace ninfer::ops::detail::hc::mix::fp8
