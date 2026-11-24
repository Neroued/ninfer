#pragma once

#include "ops/linear_attention/causal_conv/launch.h"
#include "ops/linear_attention/kernels/causal_conv.cuh"

#include <type_traits>

namespace ninfer::ops::detail {

template <class Element, bool PairsAligned>
__device__ __forceinline__ Element load_ragged_history(const __nv_bfloat16* state,
                                                       std::int64_t index) {
    if constexpr (std::is_same_v<Element, __nv_bfloat16>)
        return state[index];
    else if constexpr (PairsAligned)
        return reinterpret_cast<const __nv_bfloat162*>(state)[index];
    else
        return __halves2bfloat162(state[2 * index], state[2 * index + 1]);
}

template <class Element, bool PairsAligned>
__device__ __forceinline__ void store_ragged_history(__nv_bfloat16* state, std::int64_t index,
                                                    Element value) {
    if constexpr (std::is_same_v<Element, __nv_bfloat16>) {
        state[index] = value;
    } else if constexpr (PairsAligned) {
        reinterpret_cast<__nv_bfloat162*>(state)[index] = value;
    } else {
        state[2 * index]     = __low2bfloat16(value);
        state[2 * index + 1] = __high2bfloat16(value);
    }
}

__device__ __forceinline__ float unpack_history(__nv_bfloat16 word) {
    return __bfloat162float(word);
}
__device__ __forceinline__ float2 unpack_history(__nv_bfloat162 word) {
    return __bfloat1622float2(word);
}

__device__ __forceinline__ __nv_bfloat16 ragged_activation(float w0, float w1, float w2, float w3,
                                                          float h0, float h1, float h2, float x) {
    return causal_conv4_silu(w0, w1, w2, w3, h0, h1, h2, x);
}
__device__ __forceinline__ __nv_bfloat162
ragged_activation(float2 w0, float2 w1, float2 w2, float2 w3, float2 h0, float2 h1, float2 h2,
                  float2 x) {
    const float a = fmaf(w3.x, x.x, fmaf(w2.x, h2.x, fmaf(w1.x, h1.x, w0.x * h0.x)));
    const float b = fmaf(w3.y, x.y, fmaf(w2.y, h2.y, fmaf(w1.y, h1.y, w0.y * h0.y)));
    return __floats2bfloat162_rn(common::silu(a), common::silu(b));
}

template <int Columns, bool PairsAligned, class Element, class Output>
__global__ void causal_conv_ragged_output(const Element* x, const Element* weight,
                                          const __nv_bfloat16* source, Output output,
                                          std::int64_t channels, CausalConvRagged a) {
    __shared__ int local[Columns];
    __shared__ std::int64_t history_offset[Columns];
    const auto c = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool live = c < channels;
    const auto w0 = unpack_history(live ? weight[c] : Element{});
    const auto w1 = unpack_history(live ? weight[channels + c] : Element{});
    const auto w2 = unpack_history(live ? weight[2 * channels + c] : Element{});
    const auto w3 = unpack_history(live ? weight[3 * channels + c] : Element{});
    const auto out = output.column(live ? c : 0);

    for (auto first = std::int64_t(blockIdx.y) * Columns; first < a.tokens;
         first += std::int64_t(gridDim.y) * Columns) {
        if (threadIdx.x < Columns && first + threadIdx.x < a.tokens) {
            const auto col = first + threadIdx.x;
            const int s = a.token_to_seq[col];
            const int t = static_cast<int>(col) - a.cu_seqlens[s];
            local[threadIdx.x] = t;
            if ((threadIdx.x == 0 && t < 3) || t == 0)
                history_offset[threadIdx.x] = a.offsets.source[s] / sizeof(__nv_bfloat16);
        }
        __syncthreads();
        if (live) {
            decltype(unpack_history(Element{})) h0, h1, h2;
#pragma unroll
            for (int i = 0; i < Columns; ++i) {
                const auto col = first + i;
                if (col >= a.tokens) break;
                const int t = local[i];
                if (i == 0 || t == 0) {
                    const auto load = [&](int lag) {
                        return unpack_history(t >= lag
                            ? x[(col - lag) * channels + c]
                            : load_ragged_history<Element, PairsAligned>(
                                  source + history_offset[i], (t + 3 - lag) * channels + c));
                    };
                    h0 = load(3);
                    h1 = load(2);
                    h2 = load(1);
                }
                const auto current = unpack_history(x[col * channels + c]);
                *out.at(static_cast<int>(col)) =
                    ragged_activation(w0, w1, w2, w3, h0, h1, h2, current);
                h0 = h1;
                h1 = h2;
                h2 = current;
            }
        }
        // A later iteration may overwrite cached metadata for this iteration.
        if (first + std::int64_t(gridDim.y) * Columns < a.tokens) __syncthreads();
    }
}

template <class Element, bool PairsAligned, class Output>
__global__ void causal_conv_ragged_short(const Element* x, const Element* weight,
                                         const __nv_bfloat16* source, __nv_bfloat16* destination,
                                         Output output, std::int64_t channels, CausalConvRagged a) {
    const auto c = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const int s = blockIdx.y, begin = a.cu_seqlens[s], len = a.seq_lengths[s];
    const auto* src = source + a.offsets.source[s] / sizeof(__nv_bfloat16);
    auto* dst = destination + a.offsets.destination[s] / sizeof(__nv_bfloat16);
    const auto w0 = unpack_history(weight[c]), w1 = unpack_history(weight[channels + c]);
    const auto w2 = unpack_history(weight[2 * channels + c]);
    const auto w3 = unpack_history(weight[3 * channels + c]);
    auto h0 = load_ragged_history<Element, PairsAligned>(src, c);
    auto h1 = load_ragged_history<Element, PairsAligned>(src, channels + c);
    auto h2 = load_ragged_history<Element, PairsAligned>(src, 2 * channels + c);
    const auto out = output.column(c);
    // The same thread owns all readers and the final publication for this channel.
    for (int t = 0; t < len; ++t) {
        const auto col = begin + t;
        const auto current = x[std::int64_t(col) * channels + c];
        *out.at(col) = ragged_activation(w0, w1, w2, w3, unpack_history(h0), unpack_history(h1),
                                          unpack_history(h2), unpack_history(current));
        h0 = h1;
        h1 = h2;
        h2 = current;
    }
    store_ragged_history<Element, PairsAligned>(dst, c, h0);
    store_ragged_history<Element, PairsAligned>(dst, channels + c, h1);
    store_ragged_history<Element, PairsAligned>(dst, 2 * channels + c, h2);
}

template <class Element>
__global__ void causal_conv_ragged_state(const Element* x, const Element* source,
                                         Element* destination, std::int64_t channels,
                                         CausalConvRagged a) {
    const auto c = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    for (std::int64_t s = blockIdx.y; s < a.segments; s += gridDim.y) {
        const int n = a.seq_lengths[s], begin = a.cu_seqlens[s];
        Element h[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int p = n - 3 + j;
            h[j] = p < 0 ? source[a.offsets.source[s] / sizeof(Element) + (p + 3) * channels + c]
                          : x[(std::int64_t(begin) + p) * channels + c];
        }
        auto* dst = destination + a.offsets.destination[s] / sizeof(Element);
#pragma unroll
        for (int j = 0; j < 3; ++j) dst[j * channels + c] = h[j];
    }
}

} // namespace ninfer::ops::detail
