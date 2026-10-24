#pragma once
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"
#include "ops/common/numeric/math.cuh"

namespace ninfer::ops::detail::kda_input {
constexpr int channels = kProjectionFeatures, hidden = kInputFeatures;

__device__ void store_output(const Arguments& a, int c, int t, int b, float value) {
    auto* out = c < hidden ? a.q : c < 2 * hidden ? a.k : a.v;
    out[(static_cast<std::int64_t>(b) * a.width + t) * hidden + c % hidden] =
        __float2bfloat16_rn(value);
}

__device__ float convolution(const float* w, int c, float s0, float s1, float s2, float p) {
    float value = w[c] * s0;
    value       = fmaf(w[channels + c], s1, value);
    value       = fmaf(w[2 * channels + c], s2, value);
    return common::silu(fmaf(w[3 * channels + c], p, value));
}

template <int Width>
struct WindowEpilogue {
    Arguments a;

    template <class Output, int Columns>
    __device__ void apply_row(const Output&, int c, int first, const float (&p)[Columns],
                              int live) const {
#pragma unroll
        for (int begin = 0; begin + Width <= Columns; begin += Width) {
            if (begin + Width > live) break;
            const int b       = (first + begin) / Width;
            const int n       = a.valid ? a.valid[b] : Width;
            const auto source = a.source_offsets[b] / 2 + c;
            float h0          = __bfloat162float(a.source[source]);
            float h1          = __bfloat162float(a.source[source + channels]);
            float h2          = __bfloat162float(a.source[source + 2 * channels]);
#pragma unroll
            for (int t = 0; t < Width; ++t) {
                if (t < n) {
                    const float value = p[begin + t];
                    store_output(a, c, t, b, convolution(a.weights, c, h0, h1, h2, value));
                    if (a.records)
                        a.records[(static_cast<std::int64_t>(b) * Width + t) * channels + c] =
                            __float2bfloat16_rn(value);
                    h0 = h1;
                    h1 = h2;
                    h2 = value;
                } else
                    store_output(a, c, t, b, 0.0F);
            }
            if (a.destination) {
                const auto dest                    = a.destination_offsets[b] / 2 + c;
                a.destination[dest]                = __float2bfloat16_rn(h0);
                a.destination[dest + channels]     = __float2bfloat16_rn(h1);
                a.destination[dest + 2 * channels] = __float2bfloat16_rn(h2);
            }
        }
    }
};

struct FloatOutput {
    float* data;

    __device__ void store(int row, int column, float value) const {
        data[static_cast<std::int64_t>(column) * channels + row] = value;
    }
};

__global__ void conv_kernel(Arguments a, const float* projected) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.z;
    for (std::int64_t t0 = 4LL * blockIdx.y; t0 < a.width; t0 += 4LL * gridDim.y) {
        const int n     = a.valid ? a.valid[b] : a.width;
        const auto load = [&](std::int64_t t) {
            if (t < 0)
                return __bfloat162float(
                    a.source[a.source_offsets[b] / 2 + (t + 3LL) * channels + c]);
            return projected[(static_cast<std::int64_t>(b) * a.width + t) * channels + c];
        };
        if (t0 < n) {
            float h0 = load(t0 - 3), h1 = load(t0 - 2), h2 = load(t0 - 1);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int t = t0 + i;
                if (t < n) {
                    const float p = load(t);
                    store_output(a, c, t, b, convolution(a.weights, c, h0, h1, h2, p));
                    if (a.records)
                        a.records[(static_cast<std::int64_t>(b) * a.width + t) * channels + c] =
                            __float2bfloat16_rn(p);
                    h0 = h1;
                    h1 = h2;
                    h2 = p;
                } else if (t < a.width)
                    store_output(a, c, t, b, 0.0F);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                if (t0 + i < a.width) store_output(a, c, t0 + i, b, 0.0F);
        }
    }
}

__global__ void history_kernel(Arguments a, const float* projected) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    const int n = a.valid ? a.valid[b] : a.width;
    __nv_bfloat16 tail[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int t = n - 3 + j;
        tail[j]     = t < 0
                          ? a.source[a.source_offsets[b] / 2 + (t + 3LL) * channels + c]
                          : __float2bfloat16_rn(
                            projected[(static_cast<std::int64_t>(b) * a.width + t) * channels + c]);
    }
#pragma unroll
    for (int j = 0; j < 3; ++j)
        a.destination[a.destination_offsets[b] / 2 + static_cast<std::int64_t>(j) * channels + c] =
            tail[j];
}

} // namespace ninfer::ops::detail::kda_input
