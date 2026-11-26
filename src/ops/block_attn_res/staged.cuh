#pragma once

#include "mix.cuh"
#include "ops/common/host/launch.h"

namespace ninfer::ops::attn_res_detail {

inline constexpr int kStagedSources = 16;
inline constexpr int kStageCapacity = 72 * 1024;

template <int V>
using Packed = std::conditional_t<V == 8, uint4, Word>;

template <int V>
__device__ __forceinline__ Packed<V> load_pack(const Word* src) {
    return *reinterpret_cast<const Packed<V>*>(src);
}

template <int V>
__device__ __forceinline__ Values<V> unpack(Packed<V> packed) {
    if constexpr (V == 8)
        return decode_bf16(packed);
    else
        return {{__uint_as_float(unsigned(packed) << 16)}};
}

template <int Threads, int Q>
struct StageScratch {
    static constexpr int warps = Threads / 32;
    float stats[Q + 1][warps][kStagedSources];
    unsigned scaled[warps];
    Reduction<Threads, Q> norm;
};

template <int V>
__device__ __forceinline__ Values<V> staged_value(const Word* words, int d, int c, bool add) {
    auto value = unpack<V>(load_pack<V>(words + c));
    if (add) {
        const auto rhs = unpack<V>(load_pack<V>(words + d + c));
#pragma unroll
        for (int v = 0; v < V; ++v) value.x[v] += rhs.x[v];
    }
    return value;
}

template <int V, int Q>
__device__ __forceinline__ void accumulate_stats(const Values<V>& x, const Values<V> (&query)[Q],
                                                 float (&stats)[Q + 1]) {
#pragma unroll
    for (int v = 0; v < V; ++v) {
        stats[0] = fmaf(x.x[v], x.x[v], stats[0]);
#pragma unroll
        for (int q = 0; q < Q; ++q) stats[q + 1] = fmaf(x.x[v], query[q].x[v], stats[q + 1]);
    }
}

// Produce all source statistics and stage their represented operands. The
// convolution partial is computed across the same channels as the stack.
template <int V, int Q, int Threads, int Packs, class Partial>
__device__ __forceinline__ void stage_sources(const Mix& a, const Partial& partial,
                                              std::int64_t plane, std::int64_t col, Word* words,
                                              StageScratch<Threads, Q>& scratch) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    Values<V> query[Packs][Q]{};
#pragma unroll
    for (int i = 0; i < Packs; ++i) {
        const int c = (threadIdx.x + i * Threads) * V;
        if (c < a.d) {
#pragma unroll
            for (int q = 0; q < Q; ++q) query[i][q] = load_fp32<V>(a.queries + q * a.d + c);
        }
    }

    unsigned scaled_mask = 0;
    for (int s = 0; s <= a.stacked; ++s) {
        const bool final = s == a.stacked, add = final && a.addend;
        auto* dst       = words + s * a.d;
        const auto* src = final ? nullptr : a.stack + s * plane + col;
        Values<V> data[Packs]{};
        float stats[Q + 1]{};
#pragma unroll
        for (int i = 0; i < Packs; ++i) {
            const int c = (threadIdx.x + i * Threads) * V;
            if (c < a.d) {
                const auto bits = final ? partial.template packed<V>(c) : load_pack<V>(src + c);
                *reinterpret_cast<Packed<V>*>(dst + c) = bits;
                auto x                                 = unpack<V>(bits);
                if (add) {
                    const auto rhs                               = load_pack<V>(a.addend + col + c);
                    *reinterpret_cast<Packed<V>*>(dst + a.d + c) = rhs;
                    const auto y                                 = unpack<V>(rhs);
#pragma unroll
                    for (int v = 0; v < V; ++v) x.x[v] += y.x[v];
                }
                data[i] = x;
                accumulate_stats(x, query[i], stats);
            }
        }
#pragma unroll
        for (int k = 0; k <= Q; ++k) stats[k] = warp_sum(stats[k]);

        // Leave room for the other warps when merging their statistics.
        constexpr float limit = 0x1.fffffep+122f;
        if (!(__shfl_sync(0xffffffff, stats[0], 0) <= limit)) {
            scaled_mask |= 1u << s;
#pragma unroll
            for (int k = 0; k <= Q; ++k) stats[k] = 0.0f;
#pragma unroll
            for (int i = 0; i < Packs; ++i) {
                auto x = data[i];
#pragma unroll
                for (int v = 0; v < V; ++v) x.x[v] *= kSourceScale;
                accumulate_stats(x, query[i], stats);
            }
#pragma unroll
            for (int k = 0; k <= Q; ++k) stats[k] = warp_sum(stats[k]);
        }
        if (lane == 0) {
#pragma unroll
            for (int k = 0; k <= Q; ++k) scratch.stats[k][warp][s] = stats[k];
        }
    }
    if (lane == 0) scratch.scaled[warp] = scaled_mask;
}

template <int Q, int Threads>
__device__ __forceinline__ void
source_weights(const Mix& a, const StageScratch<Threads, Q>& scratch, float (&weights)[Q]) {
    constexpr int warps = Threads / 32;
    const int source    = threadIdx.x % kStagedSources;
    const bool live     = source <= a.stacked;
    float stats[Q + 1]{};
    float scale    = 1.0f;
    const int lane = threadIdx.x % 32;
    const unsigned exceptional =
        __reduce_or_sync(0xffffffff, lane < warps ? scratch.scaled[lane] : 0);
    if (live) {
        if (exceptional & (1u << source)) scale = kSourceScale;
#pragma unroll
        for (int w = 0; w < warps; ++w) {
            const float ratio = (scratch.scaled[w] & (1u << source)) ? 1.0f : scale;
            stats[0] += (scratch.stats[0][w][source] * ratio) * ratio;
#pragma unroll
            for (int q = 0; q < Q; ++q) stats[q + 1] += scratch.stats[q + 1][w][source] * ratio;
        }
    }
    const float inv = rms_inverse(stats[0], a.inv_d, a.params.rms_eps, scale);
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float logit = live ? (stats[q + 1] * inv) * a.params.query_scale : -CUDART_INF_F;
        float largest     = logit;
#pragma unroll
        for (int step = kStagedSources / 2; step; step /= 2)
            largest = fmaxf(largest, __shfl_xor_sync(0xffffffff, largest, step));
        const float prob  = live ? __expf(logit - largest) : 0.0f;
        float denominator = prob;
#pragma unroll
        for (int step = kStagedSources / 2; step; step /= 2)
            denominator += __shfl_xor_sync(0xffffffff, denominator, step);
        weights[q] = prob * __fdividef(1.0f, denominator);
        if (a.params.normalize_values) weights[q] *= inv * scale;
    }
}

template <int V, int Q, int Threads, int Packs, bool Norm, class Partial>
__device__ __forceinline__ void mix_staged(const Mix& a, int m, const Partial& partial, Word* words,
                                           StageScratch<Threads, Q>& scratch) {
    const auto plane = std::int64_t(a.d) * a.columns, col = std::int64_t(m) * a.d;
    stage_sources<V, Q, Threads, Packs>(a, partial, plane, col, words, scratch);
    __syncthreads();

    float weights[Q];
    source_weights(a, scratch, weights);
    Values<V> acc[Packs][Q]{};
    for (int s = 0; s <= a.stacked; ++s) {
        float coefficients[Q];
#pragma unroll
        for (int q = 0; q < Q; ++q)
            coefficients[q] =
                __shfl_sync(0xffffffff, weights[q], s + (threadIdx.x & kStagedSources));
#pragma unroll
        for (int i = 0; i < Packs; ++i) {
            const int c = (int(threadIdx.x) + i * Threads) * V;
            if (c < a.d) {
                const auto x = staged_value<V>(words + s * a.d, a.d, c, s == a.stacked && a.addend);
#pragma unroll
                for (int q = 0; q < Q; ++q) {
#pragma unroll
                    for (int v = 0; v < V; ++v)
                        acc[i][q].x[v] = fmaf(x.x[v], coefficients[q], acc[i][q].x[v]);
                }
            }
        }
    }
    finish_mix<V, Q, Threads, Packs, false, Norm>(a, plane, col, 0, acc, scratch.norm);
}

template <auto Kernel>
int prepare_stage(int bytes) {
    if (bytes > 36 * 1024) common::prepare_shared<kStageCapacity, Kernel>();
    return bytes;
}

template <class Submit>
bool dispatch_staged(const Mix& a, Submit submit) {
    if (a.stacked >= kStagedSources || a.q > 4) return false;
    // Unaligned wide columns otherwise use the tiled mix, which rereads sources
    // and handles queries separately. Staging can serve prefill for these columns.
    const bool wide_scalar = a.d % 8 != 0 && a.d > kScalarResidentChannels;
    if (a.columns > 128 && !wide_scalar) return false;
    const auto bytes = std::int64_t(a.d) * (a.stacked + 1 + (a.addend != nullptr)) * sizeof(Word);
    if (bytes > kStageCapacity) return false;

    auto geometry = [&]<int V, int Q, bool Norm>() {
        if constexpr (V == 8) {
            if (a.d <= 256 * V)
                submit.template operator()<V, Q, 256, 1, Norm>(int(bytes));
            else if (a.d <= 320 * V)
                submit.template operator()<V, Q, 320, 1, Norm>(int(bytes));
            else if (a.d <= 512 * V)
                submit.template operator()<V, Q, 256, 2, Norm>(int(bytes));
            else
                return false;
        } else {
            if (a.d <= kScalarResidentChannels)
                submit.template operator()<V, Q, 256, 2, Norm>(int(bytes));
            else if (a.d <= 3072)
                submit.template operator()<V, Q, 256, 12, Norm>(int(bytes));
            else
                return false;
        }
        return true;
    };
    auto query_count = [&]<int V, bool Norm>() {
        switch (a.q) {
        case 2:
            return geometry.template operator()<V, 2, Norm>();
        case 3:
            return geometry.template operator()<V, 3, Norm>();
        case 4:
            return geometry.template operator()<V, 4, Norm>();
        default:
            return geometry.template operator()<V, 1, Norm>();
        }
    };
    if (a.d % 8 == 0)
        return a.norm_weight ? query_count.template operator()<8, true>()
                             : query_count.template operator()<8, false>();
    return a.norm_weight ? query_count.template operator()<1, true>()
                         : query_count.template operator()<1, false>();
}

} // namespace ninfer::ops::attn_res_detail
