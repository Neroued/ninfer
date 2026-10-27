#pragma once

#include "ops/common/cuda/reduce.cuh"

#include <cuda_bf16.h>
#include <cmath>

namespace ninfer::ops::detail::qsa {

template <bool BoundedPhase>
__device__ __forceinline__ float2 prepare_sincos(double frequency, int position) {
    const double phase = double(position) * frequency;
    float sine, cosine;
    if (BoundedPhase || phase <= 2147483647.0) {
        constexpr double inv_two_pi = 0.15915494309189533577, two_pi = 6.2831853071795864769;
        const float reduced = float(phase - nearbyint(phase * inv_two_pi) * two_pi);
        __sincosf(reduced, &sine, &cosine);
    } else {
        // Non-default theta<1 can exceed the nonnegative-I32 phase range.
        double s, c;
        sincos(phase, &s, &c);
        sine   = float(s);
        cosine = float(c);
    }
    return {sine, cosine};
}

// Rescale x together with epsilon if the squared sum overflows. The caller multiplies this
// possibly rescaled x by the returned inverse, preserving the complete normalization ratio.
template <int D>
__device__ __forceinline__ float rms_inverse(float (&x)[D / 32], float epsilon) {
    float sum = 0;
#pragma unroll
    for (int j = 0; j < D / 32; ++j) sum = fmaf(x[j], x[j], sum);
    sum = common::warp_sum(sum);
    if (!isfinite(sum)) {
        float scale = 0;
#pragma unroll
        for (int j = 0; j < D / 32; ++j) scale = fmaxf(scale, fabsf(x[j]));
        scale = common::warp_max(scale);
        sum   = 0;
#pragma unroll
        for (int j = 0; j < D / 32; ++j) {
            x[j] /= scale;
            sum = fmaf(x[j], x[j], sum);
        }
        sum     = common::warp_sum(sum);
        epsilon = (epsilon / scale) / scale;
    }
    const float variance    = sum * (1.0F / D);
    const float denominator = variance + epsilon;
    // Both terms can be finite while their sum exceeds FP32. Halving preserves the ratio
    // without moving ordinary normalization to a wider accumulator.
    if (!isfinite(denominator))
        return rsqrtf(variance * 0.5F + epsilon * 0.5F) * 0.7071067811865475244F;
    return rsqrtf(denominator);
}

template <int D>
__device__ __forceinline__ void norm_rotate(float (&x)[D / 32], const __nv_bfloat16* weight,
                                            float epsilon, float2 sc, int lane) {
    const float inverse = rms_inverse<D>(x, epsilon);
#pragma unroll
    for (int j = 0; j < D / 32; ++j)
        x[j] = (x[j] * inverse) * (1.0F + __bfloat162float(weight[lane + 32 * j]));
    const float first = x[0], second = x[1];
    x[0] = first * sc.y - second * sc.x;
    x[1] = second * sc.y + first * sc.x;
}

template <int D>
__device__ __forceinline__ void prepare_head(const __nv_bfloat16* input,
                                             const __nv_bfloat16* weight, __nv_bfloat16* out,
                                             float epsilon, float2 sc, int lane) {
    float x[D / 32];
#pragma unroll
    for (int j = 0; j < D / 32; ++j) x[j] = __bfloat162float(input[lane + 32 * j]);
    norm_rotate<D>(x, weight, epsilon, sc, lane);
#pragma unroll
    for (int j = 0; j < D / 32; ++j) out[lane + 32 * j] = __float2bfloat16_rn(x[j]);
}

template <int D, bool VectorWeights>
__device__ __forceinline__ void
vector_norm_rotate(const __nv_bfloat16* input, const __nv_bfloat16* weight, float epsilon,
                   const float2* coefficients, int lane, float (&x)[D / 32]) {
    static_assert(D == 128 || D == 256);
    constexpr int N = D / 32;
    unsigned words[N / 2];
    if constexpr (N == 8) {
        const uint4 a = reinterpret_cast<const uint4*>(input)[lane];
        words[0]      = a.x;
        words[1]      = a.y;
        words[2]      = a.z;
        words[3]      = a.w;
    } else {
        const uint2 a = reinterpret_cast<const uint2*>(input)[lane];
        words[0]      = a.x;
        words[1]      = a.y;
    }
    unsigned weight_words[N / 2];
    if constexpr (VectorWeights) {
        if constexpr (N == 8) {
            const uint4 w   = reinterpret_cast<const uint4*>(weight)[lane];
            weight_words[0] = w.x;
            weight_words[1] = w.y;
            weight_words[2] = w.z;
            weight_words[3] = w.w;
        } else {
            const uint2 w   = reinterpret_cast<const uint2*>(weight)[lane];
            weight_words[0] = w.x;
            weight_words[1] = w.y;
        }
    }
#pragma unroll
    for (int j = 0; j < N; ++j)
        x[j] = __uint_as_float(((words[j / 2] >> (16 * (j % 2))) & 0xffff) << 16);
    const float inverse = rms_inverse<D>(x, epsilon);
#pragma unroll
    for (int j = 0; j < N; ++j) {
        float w;
        if constexpr (VectorWeights)
            w = __uint_as_float(((weight_words[j / 2] >> (16 * (j % 2))) & 0xffff) << 16);
        else
            w = __bfloat162float(weight[lane * N + j]);
        x[j] = (x[j] * inverse) * (1 + w);
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        const float mate = __shfl_xor_sync(0xffffffff, x[j], 32 / N);
        if (lane * N < 64) {
            const auto sc = coefficients[(lane * N + j) % 32];
            x[j]          = x[j] * sc.y + (lane * N < 32 ? -mate * sc.x : mate * sc.x);
        }
    }
}

template <int D, bool VectorWeights>
__device__ __forceinline__ void vector_head(const __nv_bfloat16* input, const __nv_bfloat16* weight,
                                            __nv_bfloat16* output, float epsilon,
                                            const float2* coefficients, int lane, bool live) {
    constexpr int N = D / 32;
    unsigned words[N / 2]{};
    if (live) {
        float x[N];
        vector_norm_rotate<D, VectorWeights>(input, weight, epsilon, coefficients, lane, x);
#pragma unroll
        for (int j = 0; j < N / 2; ++j) {
            const __nv_bfloat162_raw raw = __floats2bfloat162_rn(x[2 * j], x[2 * j + 1]);
            words[j]                     = unsigned(raw.x) | (unsigned(raw.y) << 16);
        }
    }
    if constexpr (N == 8)
        reinterpret_cast<uint4*>(output)[lane] = {words[0], words[1], words[2], words[3]};
    else
        reinterpret_cast<uint2*>(output)[lane] = {words[0], words[1]};
}
} // namespace ninfer::ops::detail::qsa
