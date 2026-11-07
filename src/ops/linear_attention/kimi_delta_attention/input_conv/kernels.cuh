#pragma once
#include "ops/linear_attention/kimi_delta_attention/input_conv/launch.h"
#include "ops/common/numeric/math.cuh"

namespace ninfer::ops::detail::kda_input {
constexpr int channels = kProjectionFeatures, hidden = kInputFeatures;

__device__ inline void store_output(const Arguments& a, int c, int t, int b, float value) {
    auto* out = c < hidden ? a.q : c < 2 * hidden ? a.k : a.v;
    out[(static_cast<std::int64_t>(b) * a.width + t) * hidden + c % hidden] =
        __float2bfloat16_rn(value);
}

template <class ConvWeight, bool FastSilu = false>
__device__ float convolution(const void* data, int c, float s0, float s1, float s2, float p) {
    const auto* w = static_cast<const ConvWeight*>(data);
    float value   = static_cast<float>(w[c]) * s0;
    value         = fmaf(static_cast<float>(w[channels + c]), s1, value);
    value         = fmaf(static_cast<float>(w[2 * channels + c]), s2, value);
    value         = fmaf(static_cast<float>(w[3 * channels + c]), p, value);
    if constexpr (FastSilu) return __fdividef(value, 1.F + __expf(-value));
    return common::silu(value);
}

template <int Width, class ConvWeight = float, bool SwizzleRows = false>
struct WindowEpilogue {
    static constexpr bool kSwizzleRowStorage = SwizzleRows;
    Arguments a;

    template <class Output, int Columns>
    __device__ void apply_row(const Output&, int c, int first, const float (&p)[Columns],
                              int live) const {
        // Width zero consumes one request whose physical extent is part of this tile.
        constexpr int window = Width ? Width : Columns;
        const int stride     = Width ? Width : a.width;
#pragma unroll
        for (int begin = 0; begin + window <= Columns; begin += window) {
            if constexpr (Width > 0)
                if (begin + Width > live) break;
            const int b       = Width ? (first + begin) / window : 0;
            const int n       = a.valid ? a.valid[b] : stride;
            const auto source = a.source_offsets[b] / 2 + c;
            float h0          = __bfloat162float(a.source[source]);
            float h1          = __bfloat162float(a.source[source + channels]);
            float h2          = __bfloat162float(a.source[source + 2 * channels]);
#pragma unroll
            for (int t = 0; t < window; ++t) {
                if (t < n) {
                    const float value = p[begin + t];
                    store_output(a, c, t, b,
                                 convolution<ConvWeight>(a.weights, c, h0, h1, h2, value));
                    if (a.records)
                        a.records[(static_cast<std::int64_t>(b) * stride + t) * channels + c] =
                            __float2bfloat16_rn(value);
                    h0 = h1;
                    h1 = h2;
                    h2 = value;
                } else if (Width > 0 || t < a.width)
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

template <int Width>
struct ParallelWindowEpilogue : WindowEpilogue<Width, __nv_bfloat16, true> {
    template <class S, class Output, class Tile>
    __device__ void finish_rows(const Output&, Tile tile, int row_begin, int column_begin, int rows,
                                int columns) const {
        const auto& a    = this->a;
        constexpr int BR = S::kBlockRows, BT = S::kBlockTokens;
        const int stride = Width ? Width : a.width;
        const int live   = min(BT, columns - column_begin);
        for (int i = threadIdx.x; i < BR * live; i += S::kThreads) {
            const int r = i % BR, local = i / BR, c = row_begin + r;
            if (c >= rows) continue;
            const int column = column_begin + local;
            const int b      = column / stride;
            const int t      = column % stride;
            const int valid  = a.valid ? a.valid[b] : stride;
            if (t >= valid) {
                store_output(a, c, t, b, 0.F);
                continue;
            }
            const float current = tile.at(r, local);
            float h[3];
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int previous = t - 3 + j;
                h[j] =
                    previous < 0
                        ? __bfloat162float(
                              a.source[a.source_offsets[b] / 2 + (previous + 3LL) * channels + c])
                        : tile.at(r, local - 3 + j);
            }
            store_output(a, c, t, b,
                         convolution<__nv_bfloat16>(a.weights, c, h[0], h[1], h[2], current));
            if (a.records)
                a.records[std::int64_t(column) * channels + c] = __float2bfloat16_rn(current);
        }
        // An in-place history publication cannot overtake any old-history reader.
        if (a.destination) {
            __syncthreads();
            const int first_request = column_begin / stride;
            const int requests      = live / stride;
            for (int i = threadIdx.x; i < BR * requests; i += S::kThreads) {
                const int r = i % BR, b = first_request + i / BR, c = row_begin + r;
                if (c >= rows) continue;
                const int valid = a.valid ? a.valid[b] : stride;
                __nv_bfloat16 history[3];
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int t = valid - 3 + j;
                    history[j] =
                        t < 0 ? a.source[a.source_offsets[b] / 2 + (t + 3LL) * channels + c]
                              : __float2bfloat16_rn(tile.at(r, b * stride + t - column_begin));
                }
#pragma unroll
                for (int j = 0; j < 3; ++j)
                    a.destination[a.destination_offsets[b] / 2 + std::int64_t(j) * channels + c] =
                        history[j];
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

template <class ConvWeight = float>
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
                    store_output(a, c, t, b, convolution<ConvWeight>(a.weights, c, h0, h1, h2, p));
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

static __global__ void history_kernel(Arguments a, const float* projected) {
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
