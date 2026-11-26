#pragma once

#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/rope/kernels/phase.cuh"
#include <cuda_bf16.h>
#include <cfloat>
#include <cstdint>

namespace ninfer::ops::detail {

// A CTA shares phases across Q and K heads of one token. Each (half) warp owns
// a complete head and retains the unrounded normalized values in registers.
template <int QueryHeads, int KeyHeads, int HeadsPerBlock, int LanesPerHead, bool WideAngles>
__global__ __launch_bounds__(HeadsPerBlock* LanesPerHead) void rmsnorm_rope_d128_kernel(
    const std::int32_t* __restrict__ positions, const __nv_bfloat162* __restrict__ q_weight,
    const __nv_bfloat162* __restrict__ k_weight, __nv_bfloat162* __restrict__ q,
    __nv_bfloat162* __restrict__ k, float epsilon, RopeD128Frequencies frequencies) {
    constexpr int Groups        = (QueryHeads + KeyHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    constexpr int Items         = 32 / LanesPerHead;
    constexpr bool CacheWeights = QueryHeads == 32 && HeadsPerBlock == 8 && LanesPerHead == 32;
    static_assert(LanesPerHead == 16 || LanesPerHead == 32);
    static_assert(HeadsPerBlock * LanesPerHead >= 64);
    const int token      = blockIdx.x / Groups;
    const int head       = (blockIdx.x % Groups) * HeadsPerBlock + threadIdx.x / LanesPerHead;
    const int lane       = threadIdx.x % LanesPerHead;
    const bool query     = head < QueryHeads;
    const int local_head = query ? head : head - QueryHeads;
    auto* data           = query ? q : k;
    const auto* weight   = query ? q_weight : k_weight;
    const auto base = (std::int64_t(token) * (query ? QueryHeads : KeyHeads) + local_head) * 64;
    __shared__ float sine[64], cosine[64];
    __shared__ __nv_bfloat162 weight_cache[CacheWeights ? 64 : 1];
    if constexpr (CacheWeights)
        if (threadIdx.x < 64) weight_cache[threadIdx.x] = weight[threadIdx.x];
    if (threadIdx.x < 64) {
        const int pair     = threadIdx.x;
        const double angle = double(positions[token]) * frequencies.value[pair];
        // theta>=1 bounds phases by the I32 position range. Keep the large-angle
        // library's register cost out of these common profiles.
        if constexpr (!WideAngles) {
            const float reduced =
                float(angle - nearbyint(angle * 0.159154943091895336) * 6.28318530717958648);
            sincosf(reduced, sine + pair, cosine + pair);
        } else {
            double s, c;
            sincos(angle, &s, &c);
            sine[pair]   = float(s);
            cosine[pair] = float(c);
        }
    }
    if constexpr (CacheWeights) __syncthreads();
    float2 first[Items]{}, second[Items]{}, gain0[Items]{}, gain1[Items]{};
    float square = 0;
    if (head < QueryHeads + KeyHeads) {
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const int i = lane + j * LanesPerHead;
            first[j]    = __bfloat1622float2(data[base + i]);
            second[j]   = __bfloat1622float2(data[base + i + 32]);
            gain0[j]    = __bfloat1622float2(CacheWeights ? weight_cache[i] : weight[i]);
            gain1[j]    = __bfloat1622float2(CacheWeights ? weight_cache[i + 32] : weight[i + 32]);
            square += first[j].x * first[j].x + first[j].y * first[j].y +
                      second[j].x * second[j].x + second[j].y * second[j].y;
        }
    }
    square        = common::warp_reduce_sum<LanesPerHead>(square);
    float inverse = lane == 0 ? rsqrtf(square * (1.F / 128.F) + epsilon) : 0.F;
    inverse       = __shfl_sync(common::kFullWarpMask, inverse, 0, LanesPerHead);
    if constexpr (!CacheWeights) __syncthreads();
    if (head < QueryHeads + KeyHeads) {
#pragma unroll
        for (int j = 0; j < Items; ++j) {
            const int i         = lane + j * LanesPerHead;
            const int pair      = i * 2;
            const float x       = first[j].x * inverse * gain0[j].x;
            const float y       = first[j].y * inverse * gain0[j].y;
            const float u       = second[j].x * inverse * gain1[j].x;
            const float v       = second[j].y * inverse * gain1[j].y;
            data[base + i]      = __floats2bfloat162_rn(x * cosine[pair] - u * sine[pair],
                                                        y * cosine[pair + 1] - v * sine[pair + 1]);
            data[base + i + 32] = __floats2bfloat162_rn(u * cosine[pair] + x * sine[pair],
                                                        v * cosine[pair + 1] + y * sine[pair + 1]);
        }
    }
}

template <int Pairs>
struct PackedBf16Pairs {
    unsigned words[Pairs];
};

template <int Pairs, bool BypassL1 = false>
__device__ __forceinline__ PackedBf16Pairs<Pairs> load_bf16_pairs(const __nv_bfloat162* ptr) {
    if constexpr (Pairs == 4) {
        const auto v = common::load_vec<uint4>(ptr);
        return {{v.x, v.y, v.z, v.w}};
    } else if constexpr (Pairs == 2) {
        const auto v =
            BypassL1 ? __ldcg(reinterpret_cast<const uint2*>(ptr)) : common::load_vec<uint2>(ptr);
        return {{v.x, v.y}};
    } else {
        return {{common::load_vec<unsigned>(ptr)}};
    }
}

template <int Pairs>
__device__ __forceinline__ void store_bf16_pairs(__nv_bfloat162* ptr,
                                                 PackedBf16Pairs<Pairs> packed) {
    if constexpr (Pairs == 4)
        common::store_vec(
            ptr, uint4{packed.words[0], packed.words[1], packed.words[2], packed.words[3]});
    else if constexpr (Pairs == 2)
        common::store_vec(ptr, uint2{packed.words[0], packed.words[1]});
    else
        common::store_vec(ptr, packed.words[0]);
}

__device__ __forceinline__ float2 expand_bf16_pair(unsigned word) {
    unsigned low, high;
    common::unpack_bf16x2_to_fp32_bits(word, low, high);
    return {__uint_as_float(low), __uint_as_float(high)};
}

__device__ __forceinline__ unsigned pack_bf16_pair(double low, double high) {
    const __nv_bfloat162_raw bits =
        __halves2bfloat162(__double2bfloat16(low), __double2bfloat16(high));
    return unsigned(bits.x) | (unsigned(bits.y) << 16);
}

template <bool WideAngles>
__device__ __forceinline__ void pair24_sincos(double angle, float& sine, float& cosine) {
    if constexpr (WideAngles) {
        double s, c;
        sincos(angle, &s, &c);
        sine   = float(s);
        cosine = float(c);
    } else {
        // theta >= 1 keeps the unreduced angle in the I32 position range.
        const float reduced =
            float(__fma_rn(-nearbyint(angle * 0.159154943091895336), 6.28318530717958648, angle));
        sine   = __sinf(reduced);
        cosine = __cosf(reduced);
    }
}

template <int Lanes, int Pairs>
__device__ __forceinline__ float pair24_inverse(const float2 (&first)[Pairs],
                                                const float2 (&second)[Pairs], float epsilon,
                                                int lane) {
    float first_sum = 0.F, second_sum = 0.F;
#pragma unroll
    for (int i = 0; i < Pairs; ++i) {
        first_sum  = fmaf(first[i].x, first[i].x, first_sum);
        second_sum = fmaf(second[i].x, second[i].x, second_sum);
        first_sum  = fmaf(first[i].y, first[i].y, first_sum);
        second_sum = fmaf(second[i].y, second[i].y, second_sum);
    }
    const float partial = common::warp_reduce_sum<Lanes>(first_sum + second_sum);
    const float sum     = __shfl_sync(common::kFullWarpMask, partial, 0, Lanes);
    const float mean    = sum * (1.F / 128.F) + epsilon;
    if (mean >= FLT_MIN && mean <= FLT_MAX) return rsqrtf(mean);

    // Finite BF16 spans a wider exponent range than FP32 can square.
    const unsigned head_mask = (common::kFullWarpMask >> (32 - Lanes))
                               << ((threadIdx.x % 32) - lane);
    double wide_sum = 0.0;
#pragma unroll
    for (int i = 0; i < Pairs; ++i) {
        wide_sum = fma(double(first[i].x), double(first[i].x), wide_sum);
        wide_sum = fma(double(first[i].y), double(first[i].y), wide_sum);
        wide_sum = fma(double(second[i].x), double(second[i].x), wide_sum);
        wide_sum = fma(double(second[i].y), double(second[i].y), wide_sum);
    }
    wide_sum = common::warp_reduce_sum<Lanes>(wide_sum, head_mask);
    wide_sum = __shfl_sync(head_mask, wide_sum, 0, Lanes);
    return float(rsqrt(wide_sum * (1.0 / 128.0) + epsilon));
}

// A CTA handles the same group of Q or K heads for one token. Threads hold the
// two split halves and their gains in registers and share a single phase table.
template <int Heads, int Lanes, bool WideAngles, bool BypassL1 = false>
__global__ __launch_bounds__(Heads* Lanes) void rmsnorm_rope_d128_pair24_kernel(
    const std::int32_t* __restrict__ positions, const __nv_bfloat162* __restrict__ q_weight,
    const __nv_bfloat162* __restrict__ k_weight, __nv_bfloat162* __restrict__ q,
    __nv_bfloat162* __restrict__ k, float epsilon,
    const __grid_constant__ RopeD128Frequencies frequencies) {
    constexpr int Groups     = 32 / Heads;
    constexpr int Pairs      = 32 / Lanes;
    constexpr bool WarpLocal = Heads * Lanes == 32;
    const int token          = blockIdx.x / Groups;
    const int head           = (blockIdx.x % Groups) * Heads + threadIdx.x / Lanes;
    const int lane           = threadIdx.x % Lanes;
    const bool query         = head < 24;
    const int local_head     = query ? head : head - 24;
    const auto base          = (std::int64_t(token) * (query ? 24 : 8) + local_head) * 64;
    auto* data               = (query ? q : k) + base + lane * Pairs;
    const auto* gain         = (query ? q_weight : k_weight) + lane * Pairs;
    const auto values0       = load_bf16_pairs<Pairs, BypassL1>(data);
    const auto values1       = load_bf16_pairs<Pairs, BypassL1>(data + 32);
    const auto gains0        = load_bf16_pairs<Pairs>(gain);
    const auto gains1        = load_bf16_pairs<Pairs>(gain + 32);

    __shared__ float sine[64], cosine[64];
    float2 lane_sine, lane_cosine;
    if constexpr (WarpLocal) {
        // A single warp produces the full phase table and exchanges it in registers.
        const int pair        = lane * (2 * Pairs) + 2 * (threadIdx.x / Lanes);
        const double position = double(positions[token]);
        pair24_sincos<WideAngles>(position * frequencies.value[pair], lane_sine.x, lane_cosine.x);
        pair24_sincos<WideAngles>(position * frequencies.value[pair + 1], lane_sine.y,
                                  lane_cosine.y);
    } else if (threadIdx.x < 64) {
        // Store each component of the vector in a separate plane. Lanes reading
        // one component address distinct banks or broadcast the same phase.
        const int pair     = (threadIdx.x % Lanes) * (2 * Pairs) + threadIdx.x / Lanes;
        const double angle = double(positions[token]) * frequencies.value[pair];
        pair24_sincos<WideAngles>(angle, sine[threadIdx.x], cosine[threadIdx.x]);
    }
    float2 first[Pairs], second[Pairs];
#pragma unroll
    for (int i = 0; i < Pairs; ++i) {
        first[i]  = expand_bf16_pair(values0.words[i]);
        second[i] = expand_bf16_pair(values1.words[i]);
    }
    const float inverse = pair24_inverse<Lanes>(first, second, epsilon, lane);
    if constexpr (!WarpLocal) __syncthreads();

    PackedBf16Pairs<Pairs> out0, out1;
#pragma unroll
    for (int i = 0; i < Pairs; ++i) {
        const auto w0 = expand_bf16_pair(gains0.words[i]);
        const auto w1 = expand_bf16_pair(gains1.words[i]);
        float s0, s1, c0, c1;
        if constexpr (WarpLocal) {
            const int source = lane + i * Lanes;
            s0               = __shfl_sync(common::kFullWarpMask, lane_sine.x, source);
            s1               = __shfl_sync(common::kFullWarpMask, lane_sine.y, source);
            c0               = __shfl_sync(common::kFullWarpMask, lane_cosine.x, source);
            c1               = __shfl_sync(common::kFullWarpMask, lane_cosine.y, source);
        } else {
            const int lo = lane + 2 * i * Lanes;
            const int hi = lo + Lanes;
            s0           = sine[lo];
            s1           = sine[hi];
            c0           = cosine[lo];
            c1           = cosine[hi];
        }
        // Keep large gains from amplifying FP32 underflow or overflowing before rotation.
        constexpr float MaxF32Gain = 0x1p14F;
        bool use_f32;
        if constexpr (WarpLocal) {
            const auto gain0 = gains0.words[i] & 0x7fff7fffU;
            const auto gain1 = gains1.words[i] & 0x7fff7fffU;
            const auto abs0  = common::load_vec<__nv_bfloat162>(&gain0);
            const auto abs1  = common::load_vec<__nv_bfloat162>(&gain1);
            use_f32 = __hgt2_mask(__hmax2(abs0, abs1), __float2bfloat162_rn(MaxF32Gain)) == 0;
        } else {
            use_f32 = fabsf(w0.x) <= MaxF32Gain && fabsf(w0.y) <= MaxF32Gain &&
                      fabsf(w1.x) <= MaxF32Gain && fabsf(w1.y) <= MaxF32Gain;
        }
        if (use_f32) {
            const float x = first[i].x * inverse * w0.x;
            const float y = first[i].y * inverse * w0.y;
            const float u = second[i].x * inverse * w1.x;
            const float v = second[i].y * inverse * w1.y;
            out0.words[i] = common::pack_bf16x2(x * c0 - u * s0, y * c1 - v * s1);
            out1.words[i] = common::pack_bf16x2(u * c0 + x * s0, v * c1 + y * s1);
        } else {
            const double x = double(first[i].x) * inverse * w0.x;
            const double y = double(first[i].y) * inverse * w0.y;
            const double u = double(second[i].x) * inverse * w1.x;
            const double v = double(second[i].y) * inverse * w1.y;
            out0.words[i]  = pack_bf16_pair(x * c0 - u * s0, y * c1 - v * s1);
            out1.words[i]  = pack_bf16_pair(u * c0 + x * s0, v * c1 + y * s1);
        }
    }
    store_bf16_pairs(data, out0);
    store_bf16_pairs(data + 32, out1);
}

} // namespace ninfer::ops::detail
