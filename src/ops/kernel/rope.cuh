#pragma once

// Implements: include/ninfer/ops/rope.h
// Fixed matches: BF16 Qwen3.6 Text 24Q/4K and 16Q/2K at D/R=256/64, DFlash 32Q/8K at
// D/R=128/128, plus packed Vision 16Q/16K at D/R=72/72. One CTA owns one token and shares its
// rotary coefficients across heads.

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

enum class RopeKernelMode : std::int32_t {
    Text1D,
    DflashText1D,
    TextMrope,
    Vision2D,
};

inline constexpr int kRopeMaxHalf = 128;

static __device__ __constant__ float kTextRopeInvFrequency[32] = {
    1.000000000e+00F, 6.042963902e-01F, 3.651741273e-01F, 2.206734069e-01F, 1.333521432e-01F,
    8.058421878e-02F, 4.869675252e-02F, 2.942727176e-02F, 1.778279410e-02F, 1.074607828e-02F,
    6.493816316e-03F, 3.924189758e-03F, 2.371373706e-03F, 1.433012570e-03F, 8.659643234e-04F,
    5.232991147e-04F, 3.162277660e-04F, 1.910952975e-04F, 1.154781985e-04F, 6.978305849e-05F,
    4.216965034e-05F, 2.548296748e-05F, 1.539926526e-05F, 9.305720409e-06F, 5.623413252e-06F,
    3.398208329e-06F, 2.053525026e-06F, 1.240937761e-06F, 7.498942093e-07F, 4.531583638e-07F,
    2.738419634e-07F, 1.654817100e-07F,
};

static __device__ __constant__ double kDflashRopeInvFrequency[64] = {
    1.00000000000000000e+00, 7.77365030238775789e-01, 6.04296390238132863e-01,
    4.69758881670649164e-01, 3.65174127254837722e-01, 2.83873596475875456e-01,
    2.20673406908458991e-01, 1.71543789634287902e-01, 1.33352143216332403e-01,
    1.03663292843769794e-01, 8.05842187761481865e-02, 6.26433536656885587e-02,
    4.86967525165863113e-02, 3.78551524925863012e-02, 2.94272717620928173e-02,
    2.28757320031839559e-02, 1.77827941003892293e-02, 1.38237222735789964e-02,
    1.07460782832131743e-02, 8.35362546957826163e-03, 6.49381631576211298e-03,
    5.04806571666747105e-03, 3.92418975848453627e-03, 3.05052789026702539e-03,
    2.37137370566165538e-03, 1.84342299240911056e-03, 1.43301257023696268e-03,
    1.11397385999480246e-03, 8.65964323360065387e-04, 6.73170382414498242e-04,
    5.23299114681494734e-04, 4.06794432108304740e-04, 3.16227766016837939e-04,
    2.45824406892019762e-04, 1.91095297497044048e-04, 1.48550801717277505e-04,
    1.15478198468945822e-04, 8.97687132447314224e-05, 6.97830584859866353e-05,
    5.42469093701132573e-05, 4.21696503428582224e-05, 3.27812115139345850e-05,
    2.54829674797934641e-05, 1.98095677855033870e-05, 1.53992652605949185e-05,
    1.19708503049572999e-05, 9.30572040929699043e-06, 7.23394162736674728e-06,
    5.62341325190349121e-06, 4.37144481261108992e-06, 3.39820832894255927e-06,
    2.64164832038609264e-06, 2.05352502645714607e-06, 1.59633854428794220e-06,
    1.24093776075171953e-06, 9.64661619911199141e-07, 7.49894209332455848e-07,
    5.82941534713607427e-07, 4.53158363760081793e-07, 3.52269465147310129e-07,
    2.73841963426436139e-07, 2.12875166179637264e-07, 1.65481709994318135e-07,
    1.28639694493697462e-07,
};

static __device__ __constant__ float kVisionRopeInvFrequency[18] = {
    1.000000000e+00F, 5.994842503e-01F, 3.593813664e-01F, 2.154434690e-01F, 1.291549665e-01F,
    7.742636827e-02F, 4.641588834e-02F, 2.782559402e-02F, 1.668100537e-02F, 1.000000000e-02F,
    5.994842503e-03F, 3.593813664e-03F, 2.154434690e-03F, 1.291549665e-03F, 7.742636827e-04F,
    4.641588834e-04F, 2.782559402e-04F, 1.668100537e-04F,
};

template <RopeKernelMode Mode>
__device__ __forceinline__ void fixed_axis_frequency(int pair, int* axis, float* frequency) {
    if constexpr (Mode == RopeKernelMode::Vision2D) {
        *axis      = pair / 18;
        *frequency = kVisionRopeInvFrequency[pair % 18];
    } else if constexpr (Mode == RopeKernelMode::DflashText1D) {
        *axis      = 0;
        *frequency = static_cast<float>(kDflashRopeInvFrequency[pair]);
    } else {
        *axis      = Mode == RopeKernelMode::TextMrope ? pair % 3 : 0;
        *frequency = kTextRopeInvFrequency[pair];
    }
}

template <RopeKernelMode Mode>
__device__ __forceinline__ void fixed_sincos(const std::int32_t* positions, int tokens, int token,
                                             int pair, float* sine, float* cosine) {
    if constexpr (Mode == RopeKernelMode::DflashText1D) {
        constexpr double kInvTwoPi = 1.59154943091895336e-01;
        constexpr double kTwoPi    = 6.28318530717958648e+00;
        const double angle  = static_cast<double>(positions[token]) * kDflashRopeInvFrequency[pair];
        const double turns  = angle * kInvTwoPi;
        const float reduced = static_cast<float>(angle - nearbyint(turns) * kTwoPi);
        sincosf(reduced, sine, cosine);
    } else {
        int axis = 0;
        float frequency;
        fixed_axis_frequency<Mode>(pair, &axis, &frequency);
        const float angle =
            static_cast<float>(positions[static_cast<std::int64_t>(axis) * tokens + token]) *
            frequency;
        sincosf(angle, sine, cosine);
    }
}

template <int HeadDim, int Half>
__device__ __forceinline__ void apply_rope_head(__nv_bfloat16* data, std::int64_t token_stride,
                                                int head, int token, int lane, float c0, float c1,
                                                float s0, float s1) {
    constexpr int kHalfPair = Half / 2;
    if (lane >= kHalfPair) { return; }
    const std::int64_t base =
        static_cast<std::int64_t>(token) * token_stride + static_cast<std::int64_t>(head) * HeadDim;
    auto* data2         = reinterpret_cast<__nv_bfloat162*>(data + base);
    const float2 first  = __bfloat1622float2(data2[lane]);
    const float2 second = __bfloat1622float2(data2[lane + kHalfPair]);
    data2[lane] = __floats2bfloat162_rn(first.x * c0 - second.x * s0, first.y * c1 - second.y * s1);
    data2[lane + kHalfPair] =
        __floats2bfloat162_rn(second.x * c0 + first.x * s0, second.y * c1 + first.y * s1);
}

template <RopeKernelMode Mode, int QHeads, int KHeads>
__global__ void rope_fixed_kernel(const std::int32_t* positions, __nv_bfloat16* q, __nv_bfloat16* k,
                                  std::int32_t tokens, std::int64_t q_token_stride,
                                  std::int64_t k_token_stride) {
    constexpr int kHeadDim = Mode == RopeKernelMode::Vision2D       ? 72
                             : Mode == RopeKernelMode::DflashText1D ? 128
                                                                    : 256;
    constexpr int kHalf    = Mode == RopeKernelMode::Vision2D       ? 36
                             : Mode == RopeKernelMode::DflashText1D ? 64
                                                                    : 32;
    const int token        = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }

    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];
    if (threadIdx.x < kHalf) {
        const int pair = static_cast<int>(threadIdx.x);
        fixed_sincos<Mode>(positions, tokens, token, pair, &sin_cache[pair], &cos_cache[pair]);
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    float c0 = 0.0F, c1 = 0.0F, s0 = 0.0F, s1 = 0.0F;
    if (lane < kHalf / 2) {
        const int pair = lane * 2;
        c0             = cos_cache[pair];
        c1             = cos_cache[pair + 1];
        s0             = sin_cache[pair];
        s1             = sin_cache[pair + 1];
    }
    for (int combined_head = warp; combined_head < QHeads + KHeads; combined_head += block_warps) {
        if (combined_head < QHeads) {
            apply_rope_head<kHeadDim, kHalf>(q, q_token_stride, combined_head, token, lane, c0, c1,
                                             s0, s1);
        } else {
            apply_rope_head<kHeadDim, kHalf>(k, k_token_stride, combined_head - QHeads, token, lane,
                                             c0, c1, s0, s1);
        }
    }
}

template <RopeKernelMode Mode, int QHeads, int KHeads, int HeadsPerBlock>
__global__ void rope_fixed_split_kernel(const std::int32_t* positions, __nv_bfloat16* q,
                                        __nv_bfloat16* k, std::int32_t tokens,
                                        std::int64_t q_token_stride, std::int64_t k_token_stride) {
    static_assert(Mode == RopeKernelMode::DflashText1D);
    constexpr int kHeadDim       = 128;
    constexpr int kHalf          = 64;
    constexpr int kCombinedHeads = QHeads + KHeads;
    constexpr int kHeadGroups    = (kCombinedHeads + HeadsPerBlock - 1) / HeadsPerBlock;
    const int token              = static_cast<int>(blockIdx.x) / kHeadGroups;
    const int head_group         = static_cast<int>(blockIdx.x) % kHeadGroups;
    if (token >= tokens) { return; }

    __shared__ float cos_cache[kHalf];
    __shared__ float sin_cache[kHalf];
    if (threadIdx.x < kHalf) {
        const int pair = static_cast<int>(threadIdx.x);
        fixed_sincos<Mode>(positions, tokens, token, pair, &sin_cache[pair], &cos_cache[pair]);
    }
    __syncthreads();

    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int local_head    = static_cast<int>(threadIdx.x) >> 5;
    const int combined_head = head_group * HeadsPerBlock + local_head;
    if (local_head >= HeadsPerBlock || combined_head >= kCombinedHeads) { return; }
    const int pair = lane * 2;
    const float c0 = cos_cache[pair];
    const float c1 = cos_cache[pair + 1];
    const float s0 = sin_cache[pair];
    const float s1 = sin_cache[pair + 1];
    if (combined_head < QHeads) {
        apply_rope_head<kHeadDim, kHalf>(q, q_token_stride, combined_head, token, lane, c0, c1, s0,
                                         s1);
    } else {
        apply_rope_head<kHeadDim, kHalf>(k, k_token_stride, combined_head - QHeads, token, lane, c0,
                                         c1, s0, s1);
    }
}

__device__ __forceinline__ void generic_axis_frequency(int axes, int head_dim, int rotary_dim,
                                                       int pair, int* axis, float* exponent) {
    if (axes == 2 && head_dim == 72 && rotary_dim == 72) {
        *axis           = pair / 18;
        const int local = pair % 18;
        *exponent       = -2.0F * static_cast<float>(local) / 36.0F;
    } else {
        *axis     = axes == 3 ? pair % 3 : 0;
        *exponent = -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
    }
}

static __global__ void rope_generic_kernel(const std::int32_t* positions, std::int32_t axes,
                                           __nv_bfloat16* q, __nv_bfloat16* k,
                                           std::int32_t head_dim, std::int32_t rotary_dim,
                                           float theta, std::int32_t q_heads, std::int32_t k_heads,
                                           std::int32_t tokens, std::int64_t q_token_stride,
                                           std::int64_t k_token_stride) {
    const int token = static_cast<int>(blockIdx.x);
    if (token >= tokens) { return; }
    const int half = rotary_dim / 2;
    __shared__ float cos_cache[kRopeMaxHalf];
    __shared__ float sin_cache[kRopeMaxHalf];
    if (threadIdx.x < static_cast<unsigned>(half)) {
        const int pair = static_cast<int>(threadIdx.x);
        if (axes == 1 && head_dim == 128 && rotary_dim == 128 && theta == 1.0e7F) {
            fixed_sincos<RopeKernelMode::DflashText1D>(positions, tokens, token, pair,
                                                       &sin_cache[pair], &cos_cache[pair]);
        } else {
            int axis       = 0;
            float exponent = 0.0F;
            generic_axis_frequency(axes, head_dim, rotary_dim, pair, &axis, &exponent);
            const float frequency = powf(theta, exponent);
            const float angle =
                static_cast<float>(positions[static_cast<std::int64_t>(axis) * tokens + token]) *
                frequency;
            sincosf(angle, &sin_cache[pair], &cos_cache[pair]);
        }
    }
    __syncthreads();

    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int block_warps = static_cast<int>(blockDim.x) >> 5;
    for (int combined_head = warp; combined_head < q_heads + k_heads;
         combined_head += block_warps) {
        const bool is_q             = combined_head < q_heads;
        const int head              = is_q ? combined_head : combined_head - q_heads;
        __nv_bfloat16* data         = is_q ? q : k;
        const std::int64_t stride_t = is_q ? q_token_stride : k_token_stride;
        const std::int64_t base     = static_cast<std::int64_t>(token) * stride_t +
                                  static_cast<std::int64_t>(head) * head_dim;
        for (int pair = lane; pair < half; pair += 32) {
            const float first        = __bfloat162float(data[base + pair]);
            const float second       = __bfloat162float(data[base + pair + half]);
            const float c            = cos_cache[pair];
            const float s            = sin_cache[pair];
            data[base + pair]        = __float2bfloat16_rn(first * c - second * s);
            data[base + pair + half] = __float2bfloat16_rn(second * c + first * s);
        }
    }
}

} // namespace ninfer::ops
