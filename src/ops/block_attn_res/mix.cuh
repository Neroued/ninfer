#pragma once

#include "launch.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <type_traits>

namespace ninfer::ops::attn_res_detail {

using Word = std::uint16_t;

template <int V>
struct Values {
    float x[V];
};

__device__ __forceinline__ Values<8> decode_bf16(const uint4& words) {
    const unsigned pairs[] = {words.x, words.y, words.z, words.w};
    Values<8> result;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        result.x[2 * i]     = __uint_as_float(pairs[i] << 16);
        result.x[2 * i + 1] = __uint_as_float(pairs[i] & 0xffff0000u);
    }
    return result;
}

template <int V>
__device__ __forceinline__ Values<V> load_bf16(const Word* src) {
    if constexpr (V == 8)
        return decode_bf16(*reinterpret_cast<const uint4*>(src));
    else
        return {{__uint_as_float(unsigned(*src) << 16)}};
}

__device__ __forceinline__ uint4 encode_bf16(const Values<8>& values) {
    unsigned pairs[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162_raw rounded =
            __floats2bfloat162_rn(values.x[2 * i], values.x[2 * i + 1]);
        pairs[i] = rounded.x | (unsigned(rounded.y) << 16);
    }
    return make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
}

template <int V>
__device__ __forceinline__ void store_bf16(Word* dst, const Values<V>& values) {
    if constexpr (V == 8)
        *reinterpret_cast<uint4*>(dst) = encode_bf16(values);
    else
        *dst = __bfloat16_as_ushort(__float2bfloat16_rn(values.x[0]));
}

template <int V>
__device__ __forceinline__ Values<V> load_fp32(const float* src) {
    if constexpr (V == 8) {
        const auto a = reinterpret_cast<const float4*>(src)[0];
        const auto b = reinterpret_cast<const float4*>(src)[1];
        return {{a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}};
    } else {
        return {{*src}};
    }
}

template <int V>
__device__ __forceinline__ void store_fp32(float* dst, const Values<V>& values) {
    if constexpr (V == 8) {
        const auto& x                     = values.x;
        reinterpret_cast<float4*>(dst)[0] = make_float4(x[0], x[1], x[2], x[3]);
        reinterpret_cast<float4*>(dst)[1] = make_float4(x[4], x[5], x[6], x[7]);
    } else {
        *dst = values.x[0];
    }
}

template <int Threads, int K>
struct Reduction {
    // Alternating buffers let the next reduction begin without waiting for
    // every warp to finish consuming the previous reduction.
    float warp_sums[2][K][Threads / 32];

    template <bool Max = false, int Count>
    __device__ __forceinline__ void reduce(float (&values)[Count], int& buffer) {
        static_assert(Count <= K);
        const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
#pragma unroll
        for (int k = 0; k < Count; ++k) {
            float x = values[k];
#pragma unroll
            for (int delta = 16; delta; delta >>= 1) {
                const float other = __shfl_down_sync(0xffffffff, x, delta);
                if constexpr (Max)
                    x = fmaxf(x, other);
                else
                    x += other;
            }
            if (lane == 0) warp_sums[buffer][k][warp] = x;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < Count; ++k) {
            float x = lane < Threads / 32 ? warp_sums[buffer][k][lane] : 0.0f;
#pragma unroll
            for (int delta = 1; delta < Threads / 32; delta <<= 1) {
                const float other = __shfl_xor_sync(0xffffffff, x, delta);
                if constexpr (Max)
                    x = fmaxf(x, other);
                else
                    x += other;
            }
            values[k] = __shfl_sync(0xffffffff, x, 0);
        }
        buffer ^= 1;
    }
};

template <int V, class Partial>
__device__ __forceinline__ Values<V> load_source(const Mix& a, const Partial& partial,
                                                 std::int64_t plane, std::int64_t col, int source,
                                                 int c) {
    if (source < a.stacked) return load_bf16<V>(a.stack + source * plane + col + c);
    auto x = partial.template load<V>(c);
    if (a.addend) {
        const auto y = load_bf16<V>(a.addend + col + c);
#pragma unroll
        for (int v = 0; v < V; ++v) x.x[v] += y.x[v];
    }
    return x;
}

// With BF16 inputs and an I32 channel count, this scale leaves headroom for
// the sum of squares even at the largest finite FP32 source value.
inline constexpr float kSourceScale = 0x1p-80f;

__device__ __forceinline__ float rms_inverse(float squares, float inv_d, float eps,
                                             float scale = 1.0f) {
    return rsqrtf(fmaf(squares, inv_d, (eps * scale) * scale));
}

template <int Q>
struct OnlineMix {
    float maximum[Q], denominator[Q];

    __device__ __forceinline__ void weights(const Mix& a, const float (&stats)[Q + 1],
                                            std::int64_t source, float (&old_weight)[Q],
                                            float (&new_weight)[Q], float scale,
                                            float scaled_inv_rms) {
        const float inv_rms = scaled_inv_rms * scale;
#pragma unroll
        for (int q = 0; q < Q; ++q) {
            const float logit = (stats[1 + q] * scaled_inv_rms) * a.params.query_scale;
            float weight = 1.0f, old = 0.0f;
            if (source != 0) {
                const float difference = logit - maximum[q];
                const float decay      = __expf(-fabsf(difference));
                old                    = difference > 0.0f ? decay : 1.0f;
                weight                 = difference > 0.0f ? 1.0f : decay;
                maximum[q]             = fmaxf(maximum[q], logit);
                denominator[q]         = fmaf(old, denominator[q], weight);
            } else {
                maximum[q]     = logit;
                denominator[q] = 1.0f;
            }
            old_weight[q] = old;
            new_weight[q] = a.params.normalize_values ? weight * inv_rms : weight;
        }
    }
};

template <int V, int Q, bool Norm>
__device__ __forceinline__ void store_mixed(const Mix& a, std::int64_t plane, std::int64_t col,
                                            int first_query, int c, const Values<V> (&acc)[Q],
                                            const float (&scales)[Q]) {
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const std::int64_t query = first_query + q;
        const auto dst           = query * plane + col + c;
        Values<V> y;
        if constexpr (Norm) {
            const auto w = load_bf16<V>(a.norm_weight + query * a.d + c);
#pragma unroll
            for (int v = 0; v < V; ++v) y.x[v] = (acc[q].x[v] * scales[q]) * w.x[v];
            store_bf16<V>(static_cast<Word*>(a.out) + dst, y);
        } else {
#pragma unroll
            for (int v = 0; v < V; ++v) y.x[v] = acc[q].x[v] * scales[q];
            store_fp32<V>(static_cast<float*>(a.out) + dst, y);
        }
    }
}

template <int V, int Q, int Threads, int Packs, bool Exact, bool Norm, class Partial>
__device__ __forceinline__ void mix_resident(const Mix& a, int m, const Partial& partial,
                                             int first_query,
                                             Reduction<Threads, Q + 1>& reduction) {
    const std::int64_t plane = std::int64_t(a.d) * a.columns, col = std::int64_t(m) * a.d;
    Values<V> queries[Packs][Q]{}, acc[Packs][Q]{};
#pragma unroll
    for (int i = 0; i < Packs; ++i) {
        const auto c = (int(threadIdx.x) + i * Threads) * V;
        if (Exact || c < a.d) {
#pragma unroll
            for (int q = 0; q < Q; ++q)
                queries[i][q] = load_fp32<V>(a.queries + std::int64_t(first_query + q) * a.d + c);
        }
    }

    OnlineMix<Q> mix{};
    int buffer = 0;
    for (std::int64_t s = 0; s <= a.stacked; ++s) {
        float stats[Q + 1]{};
        Values<V> data[Packs]{};
#pragma unroll
        for (int i = 0; i < Packs; ++i) {
            const auto c = (int(threadIdx.x) + i * Threads) * V;
            if (Exact || c < a.d) {
                data[i] = load_source<V>(a, partial, plane, col, int(s), c);
#pragma unroll
                for (int v = 0; v < V; ++v) {
                    const float x = data[i].x[v];
                    stats[0]      = fmaf(x, x, stats[0]);
#pragma unroll
                    for (int q = 0; q < Q; ++q)
                        stats[q + 1] = fmaf(x, queries[i][q].x[v], stats[q + 1]);
                }
            }
        }
        reduction.reduce(stats, buffer);
        float source_scale = 1.0f, eps = a.params.rms_eps;
        if (!isfinite(stats[0])) {
            // Reuse the same register-resident source across the scaled reduction.
            source_scale = kSourceScale;
            eps          = (eps * kSourceScale) * kSourceScale;
#pragma unroll
            for (int q = 0; q <= Q; ++q) stats[q] = 0.0f;
#pragma unroll
            for (int i = 0; i < Packs; ++i) {
#pragma unroll
                for (int v = 0; v < V; ++v) {
                    const float x = data[i].x[v] * kSourceScale;
                    stats[0]      = fmaf(x, x, stats[0]);
#pragma unroll
                    for (int q = 0; q < Q; ++q)
                        stats[q + 1] = fmaf(x, queries[i][q].x[v], stats[q + 1]);
                }
            }
            reduction.reduce(stats, buffer);
        }
        float old_w[Q], new_w[Q];
        mix.weights(a, stats, s, old_w, new_w, source_scale, rms_inverse(stats[0], a.inv_d, eps));
        if constexpr (Norm) {
            // Divide the final source update by the softmax denominator.
            // The values entering RMSNorm are the mix, with no wide numerator.
            if (s == a.stacked) {
#pragma unroll
                for (int q = 0; q < Q; ++q) {
                    const float inv_den = __frcp_rn(mix.denominator[q]);
                    old_w[q] *= inv_den;
                    new_w[q] *= inv_den;
                }
            }
        }
#pragma unroll
        for (int i = 0; i < Packs; ++i) {
#pragma unroll
            for (int q = 0; q < Q; ++q) {
#pragma unroll
                for (int v = 0; v < V; ++v)
                    acc[i][q].x[v] = fmaf(data[i].x[v], new_w[q], acc[i][q].x[v] * old_w[q]);
            }
        }
    }

    float scales[Q];
    if constexpr (Norm) {
        float stats[Q]{};
#pragma unroll
        for (int q = 0; q < Q; ++q) {
#pragma unroll
            for (int i = 0; i < Packs; ++i) {
#pragma unroll
                for (int v = 0; v < V; ++v) {
                    const float value = acc[i][q].x[v];
                    stats[q]          = fmaf(value, value, stats[q]);
                }
            }
        }
        reduction.reduce(stats, buffer);
        bool overflow = false;
#pragma unroll
        for (int q = 0; q < Q; ++q) overflow |= !isfinite(stats[q]);
        if (overflow) {
            float max_abs[Q]{};
#pragma unroll
            for (int q = 0; q < Q; ++q) {
#pragma unroll
                for (int i = 0; i < Packs; ++i) {
#pragma unroll
                    for (int v = 0; v < V; ++v)
                        max_abs[q] = fmaxf(max_abs[q], fabsf(acc[i][q].x[v]));
                }
            }
            reduction.template reduce<true>(max_abs, buffer);
            float scaled_squares[Q]{};
#pragma unroll
            for (int q = 0; q < Q; ++q) {
                scales[q] = isfinite(stats[q]) ? 1.0f : __frcp_rn(max_abs[q]);
#pragma unroll
                for (int i = 0; i < Packs; ++i) {
#pragma unroll
                    for (int v = 0; v < V; ++v) {
                        const float value = acc[i][q].x[v] * scales[q];
                        acc[i][q].x[v]    = value;
                        scaled_squares[q] = fmaf(value, value, scaled_squares[q]);
                    }
                }
            }
            reduction.reduce(scaled_squares, buffer);
#pragma unroll
            for (int q = 0; q < Q; ++q)
                scales[q] = rms_inverse(scaled_squares[q], a.inv_d, a.params.norm_eps, scales[q]);
        } else {
#pragma unroll
            for (int q = 0; q < Q; ++q)
                scales[q] = rms_inverse(stats[q], a.inv_d, a.params.norm_eps);
        }
    } else {
#pragma unroll
        for (int q = 0; q < Q; ++q) scales[q] = __frcp_rn(mix.denominator[q]);
    }

#pragma unroll
    for (int i = 0; i < Packs; ++i) {
        const auto c = (int(threadIdx.x) + i * Threads) * V;
        if (Exact || c < a.d)
            store_mixed<V, Q, Norm>(a, plane, col, first_query, c, acc[i], scales);
    }
}

// One CTA caches scalar source coefficients for ordinary histories. Very large
// histories can still use the streaming route without caller-owned scratch.
inline constexpr int kCachedSources = 256;

template <int Threads>
struct GeneralScratch {
    Reduction<Threads, 2> reduction;
    float2 coefficients[kCachedSources];
};

template <int V, class Partial>
__device__ __forceinline__ void scan_source(const Mix& a, const Partial& partial,
                                            std::int64_t plane, std::int64_t col,
                                            const float* query, int source, std::int64_t tile,
                                            float scale, Values<V>& selected, float (&stats)[2]) {
    for (std::int64_t offset = 0; offset < a.d; offset += blockDim.x * V) {
        const auto c = offset + threadIdx.x * V;
        if (c >= a.d) continue;
        const auto x = load_source<V>(a, partial, plane, col, source, int(c));
        const auto w = load_fp32<V>(query + c);
        if (offset == tile) selected = x;
#pragma unroll
        for (int v = 0; v < V; ++v) {
            const float value = x.x[v] * scale;
            stats[0]          = fmaf(value, value, stats[0]);
            stats[1]          = fmaf(value, w.x[v], stats[1]);
        }
    }
}

template <int V, int Threads, class Partial>
__device__ __forceinline__ float
source_stats(const Mix& a, const Partial& partial, std::int64_t plane, std::int64_t col,
             const float* query, int source, std::int64_t tile, Values<V>& selected,
             float (&stats)[2], Reduction<Threads, 2>& reduction, int& buffer) {
    scan_source(a, partial, plane, col, query, source, tile, 1.0f, selected, stats);
    reduction.reduce(stats, buffer);
    if (isfinite(stats[0])) return 1.0f;
    stats[0] = 0.0f;
    stats[1] = 0.0f;
    scan_source(a, partial, plane, col, query, source, tile, kSourceScale, selected, stats);
    reduction.reduce(stats, buffer);
    return kSourceScale;
}

template <int V, int Threads, class Partial>
__device__ __forceinline__ void
cache_coefficients(const Mix& a, const Partial& partial, std::int64_t plane, std::int64_t col,
                   const float* query, GeneralScratch<Threads>& scratch, int& buffer) {
    float max_logit = -CUDART_INF_F;
    for (int s = 0; s <= a.stacked; ++s) {
        Values<V> ignored;
        float stats[2]{};
        const float scale   = source_stats(a, partial, plane, col, query, s, -1, ignored, stats,
                                           scratch.reduction, buffer);
        const float inv_rms = rms_inverse(stats[0], a.inv_d, a.params.rms_eps, scale);
        const float logit   = (stats[1] * inv_rms) * a.params.query_scale;
        max_logit           = fmaxf(max_logit, logit);
        if (threadIdx.x == 0)
            scratch.coefficients[s] =
                make_float2(logit, a.params.normalize_values ? inv_rms * scale : 1.0f);
    }
    __syncthreads();

    float denominator[2]{};
    for (int s = threadIdx.x; s <= a.stacked; s += Threads) {
        auto& coeff             = scratch.coefficients[s];
        const float probability = __expf(coeff.x - max_logit);
        denominator[0] += probability;
        coeff.x = probability * coeff.y;
    }
    scratch.reduction.reduce(denominator, buffer);
    const float inv_den = __frcp_rn(denominator[0]);
    for (int s = threadIdx.x; s <= a.stacked; s += Threads) scratch.coefficients[s].x *= inv_den;
    __syncthreads();
}

template <int V, int Threads, bool Norm, bool Cached, class Partial>
__device__ __forceinline__ void mix_tiled(const Mix& a, int m, const Partial& partial, int q,
                                          GeneralScratch<Threads>& scratch) {
    const std::int64_t plane = std::int64_t(a.d) * a.columns, col = std::int64_t(m) * a.d;
    const auto* query = a.queries + std::int64_t(q) * a.d;
    int buffer        = 0;
    if constexpr (Cached) cache_coefficients<V>(a, partial, plane, col, query, scratch, buffer);

    float input_scale = 1.0f, scale[1]{1.0f};
    // Pass 0 finds the output RMS. Pass 1 is only needed when the sum of
    // squares overflows. Pass 2 emits the result.
    for (int pass = Norm ? 0 : 2; pass <= 2; ++pass) {
        float squares[2]{};
        float max_abs[2]{};
        for (std::int64_t tile = 0; tile < a.d; tile += Threads * V) {
            const auto c = tile + threadIdx.x * V;
            Values<V> acc[1]{};
            if constexpr (Cached) {
                if (c < a.d) {
                    for (int s = 0; s <= a.stacked; ++s) {
                        const auto x      = load_source<V>(a, partial, plane, col, s, int(c));
                        const float coeff = scratch.coefficients[s].x;
#pragma unroll
                        for (int v = 0; v < V; ++v) acc[0].x[v] = fmaf(x.x[v], coeff, acc[0].x[v]);
                    }
                }
            } else {
                OnlineMix<1> mix{};
                for (std::int64_t s = 0; s <= a.stacked; ++s) {
                    Values<V> selected{};
                    float stats[2]{};
                    const float source_scale =
                        source_stats(a, partial, plane, col, query, int(s), tile, selected, stats,
                                     scratch.reduction, buffer);
                    float old_w[1], new_w[1];
                    mix.weights(a, stats, s, old_w, new_w, source_scale,
                                rms_inverse(stats[0], a.inv_d, a.params.rms_eps, source_scale));
#pragma unroll
                    for (int v = 0; v < V; ++v)
                        acc[0].x[v] = fmaf(selected.x[v], new_w[0], acc[0].x[v] * old_w[0]);
                }
                const float inv_den = __frcp_rn(mix.denominator[0]);
#pragma unroll
                for (int v = 0; v < V; ++v) acc[0].x[v] *= inv_den;
            }
            if (c < a.d) {
#pragma unroll
                for (int v = 0; v < V; ++v) acc[0].x[v] *= input_scale;
                if (pass < 2) {
#pragma unroll
                    for (int v = 0; v < V; ++v) {
                        const float value = acc[0].x[v];
                        squares[0]        = fmaf(value, value, squares[0]);
                        max_abs[0]        = fmaxf(max_abs[0], fabsf(value));
                    }
                } else {
                    store_mixed<V, 1, Norm>(a, plane, col, q, int(c), acc, scale);
                }
            }
        }
        if (pass < 2) {
            scratch.reduction.reduce(squares, buffer);
            if (pass == 0 && !isfinite(squares[0])) {
                scratch.reduction.template reduce<true>(max_abs, buffer);
                input_scale = __frcp_rn(max_abs[0]);
            } else {
                scale[0] = rms_inverse(squares[0], a.inv_d, a.params.norm_eps, input_scale);
                pass     = 1;
            }
        }
    }
}

template <int V, int Threads, bool Norm, class Partial>
__device__ __forceinline__ void mix_general(const Mix& a, int m, const Partial& partial, int q,
                                            GeneralScratch<Threads>& scratch) {
    if (a.stacked < kCachedSources)
        mix_tiled<V, Threads, Norm, true>(a, m, partial, q, scratch);
    else
        mix_tiled<V, Threads, Norm, false>(a, m, partial, q, scratch);
}

template <int Threads, int Q, int Packs>
using MixScratch =
    std::conditional_t<Packs != 0, Reduction<Threads, Q + 1>, GeneralScratch<Threads>>;

template <int V, bool Norm>
__device__ __forceinline__ void zero_mix(const Mix& a, int m) {
    const std::int64_t plane = std::int64_t(a.d) * a.columns, col = std::int64_t(m) * a.d;
    for (std::int64_t c = threadIdx.x * V; c < a.d; c += blockDim.x * V) {
        for (int q = 0; q < a.q; ++q) {
            const auto dst = q * plane + col + c;
            if constexpr (Norm)
                store_bf16<V>(static_cast<Word*>(a.out) + dst, {});
            else
                store_fp32<V>(static_cast<float*>(a.out) + dst, {});
        }
    }
}

// Select solely on representation, query register footprint and channel geometry.
template <class Submit>
void dispatch_mix(const Mix& a, Submit submit) {
    auto geometry = [&]<int Q, bool Norm>() {
        if (a.d == 2304)
            submit.template operator()<8, Q, 288, 1, true, Norm>();
        else if (a.d % 8 == 0 && a.d <= 4096)
            submit.template operator()<8, Q, 256, 2, false, Norm>();
        else if (a.d % 8 != 0 && a.d <= 512)
            submit.template operator()<1, Q, 128, 4, false, Norm>();
        else if (a.d % 8 == 0)
            submit.template operator()<8, 1, 128, 0, false, Norm>();
        else
            submit.template operator()<1, 1, 128, 0, false, Norm>();
    };
    auto queries = [&]<bool Norm>() {
        switch (a.q) {
        case 2:
            geometry.template operator()<2, Norm>();
            break;
        case 3:
            geometry.template operator()<3, Norm>();
            break;
        case 4:
            geometry.template operator()<4, Norm>();
            break;
        default:
            geometry.template operator()<1, Norm>();
            break;
        }
    };
    if (a.norm_weight)
        queries.template operator()<true>();
    else
        queries.template operator()<false>();
}

} // namespace ninfer::ops::attn_res_detail
