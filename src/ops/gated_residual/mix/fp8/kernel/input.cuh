#pragma once
#include "ops/gated_residual/mix/fp8/operands.h"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/math.cuh"
#include <cuda_fp8.h>

namespace ninfer::ops::detail::hc::mix::fp8 {
// Encode the down input and compute all four BF16/A16 injection dots from the
// same original values. The encoded activation is never used by injection.
__global__ __launch_bounds__(256) void encode_injection(const __nv_bfloat16* input,
                                                        const __nv_bfloat16* weight,
                                                        std::uint8_t* codes, float* scales,
                                                        float* injection) {
    constexpr int Pairs = kTotal / 2, Slots = Pairs / 256;
    const auto m  = std::int64_t(blockIdx.x);
    const int tid = threadIdx.x;
    float2 values[Slots];
    float dots[4]{}, maximum = 0;
    const auto accumulate = [&](int p, float2 x) {
        maximum = fmaxf(maximum, fmaxf(fabsf(x.x), fabsf(x.y)));
#pragma unroll
        for (int stream = 0; stream < 4; ++stream) {
            auto w = __bfloat1622float2(
                reinterpret_cast<const __nv_bfloat162*>(weight)[stream * Pairs + p]);
            dots[stream] = fmaf(x.x, w.x, dots[stream]);
            dots[stream] = fmaf(x.y, w.y, dots[stream]);
        }
    };
#pragma unroll
    for (int v = 0; v < Slots; ++v) {
        const int p = tid + 256 * v;
        values[v] =
            __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(input)[m * Pairs + p]);
        accumulate(p, values[v]);
    }
    __shared__ float sums[4][8];
    const float bound = common::block_max_broadcast<256>(maximum);
    const float scale = bound > 0 ? bound / 448.F : 0.F, inv = scale > 0 ? 1.F / scale : 0.F;
#pragma unroll
    for (int s = 0; s < 4; ++s) {
        const float dot = common::block_reduce_sum<256>(dots[s], sums[s]);
        if (tid == 0) injection[m * 4 + s] = 2.F * common::sigmoid(dot * .25F);
    }
#pragma unroll
    for (int v = 0; v < Slots; ++v) {
        int p        = tid + 256 * v;
        const auto x = values[v];
        reinterpret_cast<std::uint16_t*>(codes)[m * Pairs + p] =
            __nv_cvt_float2_to_fp8x2(make_float2(x.x * inv, x.y * inv), __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) scales[m] = scale;
}

// Materialize complete MMA token tiles in private codes/scales. Padding does
// not read BF16 u and is never published as an output column.
__global__ __launch_bounds__(256) void encode_up(const __nv_bfloat16* input, std::uint8_t* codes,
                                                 float* scales, int columns) {
    constexpr int Pairs = kRank / 2, Slots = (Pairs + 255) / 256;
    const int token = blockIdx.x, tid = threadIdx.x;
    if (token >= columns) {
        for (int pair = tid; pair < Pairs; pair += 256)
            reinterpret_cast<std::uint16_t*>(codes)[std::int64_t(token) * Pairs + pair] = 0;
        if (tid == 0) scales[token] = 0;
        return;
    }
    float2 values[Slots];
    float maximum = 0;
#pragma unroll
    for (int item = 0; item < Slots; ++item) {
        const int pair = tid + item * 256;
        values[item]   = pair < Pairs ? __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(
                                          input)[std::int64_t(token) * Pairs + pair])
                                      : make_float2(0, 0);
        maximum        = fmaxf(maximum, fmaxf(fabsf(values[item].x), fabsf(values[item].y)));
    }
    const float bound = common::block_max_broadcast<256>(maximum);
    const float scale = bound > 0 ? bound / 448.F : 0.F, inv = scale > 0 ? 1.F / scale : 0.F;
#pragma unroll
    for (int item = 0; item < Slots; ++item) {
        const int pair = tid + item * 256;
        if (pair < Pairs)
            reinterpret_cast<std::uint16_t*>(codes)[std::int64_t(token) * Pairs + pair] =
                __nv_cvt_float2_to_fp8x2(make_float2(values[item].x * inv, values[item].y * inv),
                                         __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) scales[token] = scale;
}
} // namespace ninfer::ops::detail::hc::mix::fp8
