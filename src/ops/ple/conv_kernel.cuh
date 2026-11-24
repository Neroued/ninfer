#pragma once

#include "ops/ple/operands.h"
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail::ple {

struct ConvArguments {
    const __nv_bfloat16 *residual, *gated, *kernel, *history, *records, *norm;
    __nv_bfloat16 *updated, *normalized;
    const std::int32_t *valid, *offsets;
    const std::int64_t* state_offsets;
    std::int64_t history_column, record_column, record_batch;
    int tokens, batch;
    float epsilon;
    const std::int32_t *cu_seqlens = nullptr, *token_to_seq = nullptr;
};

struct ConvWindow {
    const __nv_bfloat16 *history, *records;
    std::int64_t history_column, record_column;

    __device__ const __nv_bfloat16* at(int channel, int position) const {
        return position < 0 ? history + (position + kHistory) * history_column + channel
                            : records + position * record_column + channel;
    }
};

template <bool Ragged>
__device__ inline ConvWindow window(const ConvArguments& a, int token, int seq, int start = 0) {
    const auto first = Ragged ? std::int64_t(start) * a.record_column
                              : seq * a.record_batch + a.offsets[seq] * a.record_column;
    return {token < kHistory ? a.history + a.state_offsets[seq] / 2 : nullptr, a.records + first,
            a.history_column, a.record_column};
}

template <int Width>
__device__ common::Bf16Vector<Width> conv_value(const ConvArguments& a, const ConvWindow& source,
                                                std::int64_t column, int token, int channel) {
    float conv[Width]{};
#pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
        const auto n = common::Bf16Vector<Width>::load(source.at(channel, token - 9 + tap * 3));
        const auto weight = common::Bf16Vector<Width>::load(a.kernel + tap * kChannels + channel);
#pragma unroll
        for (int v = 0; v < Width; ++v)
            conv[v] = fmaf(__bfloat162float(weight.word[v]), __bfloat162float(n.word[v]), conv[v]);
    }
    const auto offset   = column * kChannels + channel;
    const auto residual = common::Bf16Vector<Width>::load(a.residual + offset);
    const auto gated    = common::Bf16Vector<Width>::load(a.gated + offset);
    common::Bf16Vector<Width> out;
#pragma unroll
    for (int v = 0; v < Width; ++v)
        out.word[v] = __float2bfloat16_rn(__bfloat162float(residual.word[v]) +
                                          __bfloat162float(gated.word[v]) + common::silu(conv[v]));
    return out;
}

template <int Block, int Width, bool Ragged>
__global__ void conv_add_kernel(const __grid_constant__ ConvArguments a) {
    constexpr int tiles     = (kChannels + Block * Width - 1) / (Block * Width);
    const std::int64_t work = std::int64_t(a.tokens) * a.batch * tiles;
    for (std::int64_t job = blockIdx.x; job < work; job += gridDim.x) {
        const auto column = job / tiles;
        const int seq     = Ragged ? a.token_to_seq[column] : column / a.tokens;
        const int start   = Ragged ? a.cu_seqlens[seq] : 0;
        const int t       = Ragged ? column - start : column % a.tokens;
        const int c       = (job % tiles * Block + threadIdx.x) * Width;
        if (c >= kChannels) continue;
        common::Bf16Vector<Width> result{};
        if (Ragged || t < a.valid[seq])
            result = conv_value<Width>(a, window<Ragged>(a, t, seq, start), column, t, c);
        result.store(a.updated + column * kChannels + c);
    }
}

template <int Width>
__device__ __forceinline__ void load_history(float (&history)[kHistory][Width],
                                             const ConvWindow& source, int c, int t, int rotation) {
#pragma unroll
    for (int j = 0; j < kHistory; ++j) {
        const auto n = common::Bf16Vector<Width>::load(source.at(c, t - kHistory + j));
#pragma unroll
        for (int v = 0; v < Width; ++v)
            history[(rotation + j) % kHistory][v] = __bfloat162float(n.word[v]);
    }
}

template <int Block, int Width, int Tile, bool Ragged>
__global__ void conv_add_tiled_kernel(const __grid_constant__ ConvArguments a) {
    constexpr int channels = (kChannels + Block * Width - 1) / (Block * Width);
    const auto time_tiles  = (std::int64_t(a.tokens) + Tile - 1) / Tile;
    const auto work        = time_tiles * a.batch * channels;
    for (std::int64_t job = blockIdx.x; job < work; job += gridDim.x) {
        const int c = (job % channels * Block + threadIdx.x) * Width;
        if (c >= kChannels) continue;
        const int b     = Ragged ? 0 : job / channels / time_tiles;
        const int first = (Ragged ? job / channels : job / channels % time_tiles) * Tile;
        const int valid = Ragged ? a.tokens : a.valid[b];
        float history[kHistory][Width], weight[4][Width];
        ConvWindow source{};
        if (first < valid) {
            if constexpr (!Ragged) {
                source = window<false>(a, first, b);
                load_history(history, source, c, first, 0);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const auto w = common::Bf16Vector<Width>::load(a.kernel + j * kChannels + c);
#pragma unroll
                for (int v = 0; v < Width; ++v) weight[j][v] = __bfloat162float(w.word[v]);
            }
        }
#pragma unroll
        for (int i = 0; i < Tile; ++i) {
            const auto column = std::int64_t(first) + i;
            if (column >= a.tokens) break;
            const auto out = (std::int64_t(b) * a.tokens + column) * kChannels + c;
            common::Bf16Vector<Width> y{};
            if (Ragged || column < valid) {
                int t = column;
                if constexpr (Ragged) {
                    const int seq   = a.token_to_seq[column];
                    const int start = a.cu_seqlens[seq];
                    t               = column - start;
                    if (i == 0 || t == 0) {
                        source = window<true>(a, t, seq, start);
                        // Align the new segment's history with the current ring index.
                        load_history(history, source, c, t, i);
                    }
                }
                const auto n = common::Bf16Vector<Width>::load(source.at(c, t));
                const auto h = common::Bf16Vector<Width>::load(a.residual + out);
                const auto u = common::Bf16Vector<Width>::load(a.gated + out);
#pragma unroll
                for (int v = 0; v < Width; ++v) {
                    float sum = fmaf(weight[0][v], history[i % 9][v], 0.f);
                    sum       = fmaf(weight[1][v], history[(i + 3) % 9][v], sum);
                    sum       = fmaf(weight[2][v], history[(i + 6) % 9][v], sum);
                    sum       = fmaf(weight[3][v], __bfloat162float(n.word[v]), sum);
                    y.word[v] =
                        __float2bfloat16_rn(__bfloat162float(h.word[v]) +
                                            __bfloat162float(u.word[v]) + common::silu(sum));
                    history[i % 9][v] = __bfloat162float(n.word[v]);
                }
            }
            y.store(a.updated + out);
        }
    }
}

// Each CTA owns a stream/token. Only rounded Y participates in the norm; the
// same words are written to updated and retained in registers for the reduction.
template <int Block, int Width, bool WithConv, bool Ragged>
__global__ void conv_norm_kernel(const __grid_constant__ ConvArguments a) {
    constexpr int items = (kFeatures + Block * Width - 1) / (Block * Width);
    __shared__ float sums[Block / 32];
    __shared__ float inverse;
    const std::int64_t work = std::int64_t(a.tokens) * a.batch * kStreams;
    for (std::int64_t job = blockIdx.x; job < work; job += gridDim.x) {
        const auto column = job / kStreams;
        const int s       = job % kStreams;
        int t             = 0;
        ConvWindow source{};
        if constexpr (Ragged) {
            if constexpr (WithConv) {
                const int seq   = a.token_to_seq[column];
                const int start = a.cu_seqlens[seq];
                t               = column - start;
                source          = window<true>(a, t, seq, start);
            }
        } else {
            const int b = column / a.tokens;
            t           = column % a.tokens;
            if (t >= a.valid[b]) {
#pragma unroll
                for (int i = 0; i < items; ++i) {
                    const int d = (threadIdx.x + i * Block) * Width;
                    if (d < kFeatures) {
                        const auto offset = column * kChannels + s * kFeatures + d;
                        if constexpr (WithConv)
                            common::Bf16Vector<Width>{}.store(a.updated + offset);
                        common::Bf16Vector<Width>{}.store(a.normalized + offset);
                    }
                }
                continue;
            }
            if constexpr (WithConv) source = window<false>(a, t, b);
        }
        common::Bf16Vector<Width> values[items];
        float sum = 0;
#pragma unroll
        for (int i = 0; i < items; ++i) {
            const int d = (threadIdx.x + i * Block) * Width;
            if (d < kFeatures) {
                const int c = s * kFeatures + d;
                if constexpr (WithConv) {
                    values[i] = conv_value<Width>(a, source, column, t, c);
                    values[i].store(a.updated + column * kChannels + c);
                } else
                    values[i] = common::Bf16Vector<Width>::load(a.updated + column * kChannels + c);
#pragma unroll
                for (int v = 0; v < Width; ++v) {
                    const float y = __bfloat162float(values[i].word[v]);
                    sum           = fmaf(y, y, sum);
                }
            }
        }
        sum = common::block_reduce_sum<Block>(sum, sums);
        if (threadIdx.x == 0) inverse = rsqrtf(sum / kFeatures + a.epsilon);
        __syncthreads();
        const float inv = inverse;
#pragma unroll
        for (int i = 0; i < items; ++i) {
            const int d = (threadIdx.x + i * Block) * Width;
            if (d < kFeatures) {
                const int c       = s * kFeatures + d;
                const auto weight = common::Bf16Vector<Width>::load(a.norm + c);
                common::Bf16Vector<Width> out;
#pragma unroll
                for (int v = 0; v < Width; ++v)
                    out.word[v] = __float2bfloat16_rn(__bfloat162float(values[i].word[v]) * inv *
                                                      (1.f + __bfloat162float(weight.word[v])));
                out.store(a.normalized + column * kChannels + c);
            }
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops::detail::ple
