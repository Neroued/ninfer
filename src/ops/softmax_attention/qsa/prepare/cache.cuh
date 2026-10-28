#pragma once

#include "ninfer/types.h"
#include "ops/common/quant/fp8.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/common/quant/integer.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/softmax_attention/qsa/cache_format.h"
#include "ops/softmax_attention/qsa/prepare/norm_rope.cuh"
#include "ops/softmax_attention/qsa/prepare/arguments.cuh"

namespace ninfer::ops::detail::qsa {

__device__ __forceinline__ void store_int8(const float (&values)[8], std::uint8_t* data,
                                           std::uint8_t* scales, int lane) {
    auto* codes      = reinterpret_cast<std::int8_t*>(data);
    auto* out_scales = reinterpret_cast<__half*>(scales);
#pragma unroll
    for (int group = 0; group < 4; ++group) {
        const float x0 = values[2 * group], x1 = values[2 * group + 1];
        const auto quant = common::int8_fp16_scale(common::warp_max(fmaxf(fabsf(x0), fabsf(x1))));
        codes[group * 64 + lane]      = common::int8_quant_code(x0, quant.inverse_scale);
        codes[group * 64 + lane + 32] = common::int8_quant_code(x1, quant.inverse_scale);
        if (lane == 0) out_scales[group] = quant.scale;
    }
}

__device__ __forceinline__ void store_fp8(const float (&values)[8], std::uint8_t* codes,
                                          std::uint8_t* scales, int lane) {
    float absmax = 0;
#pragma unroll
    for (float value : values) absmax = fmaxf(absmax, fabsf(value));
    const auto quant = common::fp8_fp16_scale(common::warp_max(absmax));
#pragma unroll
    for (int j = 0; j < 8; ++j)
        codes[lane + 32 * j] = common::fp8_quant_code(values[j], quant.inverse_scale);
    if (lane == 0) *reinterpret_cast<__half*>(scales) = quant.scale;
}

__device__ __forceinline__ void store_nvfp4(const float (&values)[8], std::uint8_t* codes,
                                            std::uint8_t* scales, int lane, float* scratch) {
#pragma unroll
    for (int j = 0; j < 8; ++j) scratch[lane + 32 * j] = values[j];
    __syncwarp();
    if (lane < 16) {
        const auto encoded = common::quantize_nvfp4_group16_f32(scratch + 16 * lane);
        auto* out          = codes + 8 * lane;
        if (reinterpret_cast<std::uintptr_t>(out) % 8 == 0)
            *reinterpret_cast<uint2*>(out) = make_uint2(encoded.codes_lo, encoded.codes_hi);
        else {
            // Quantized planes promise natural byte alignment, including padded layer views.
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                out[j]     = std::uint8_t(encoded.codes_lo >> (8 * j));
                out[j + 4] = std::uint8_t(encoded.codes_hi >> (8 * j));
            }
        }
        scales[lane] = encoded.scale;
    }
    // Every reader must finish before the warp reuses this row for V.
    __syncwarp();
}

template <KvCacheStorage Storage>
__device__ __forceinline__ void store_quantized(const float (&values)[8], std::uint8_t* codes,
                                                std::uint8_t* scales, int lane, float* scratch) {
    if constexpr (Storage == KvCacheStorage::Int8Group64)
        store_int8(values, codes, scales, lane);
    else if constexpr (Storage == KvCacheStorage::Fp8E4M3Row256)
        store_fp8(values, codes, scales, lane);
    else {
        static_assert(Storage == KvCacheStorage::Nvfp4Group16);
        store_nvfp4(values, codes, scales, lane, scratch);
    }
}

template <KvCacheStorage Storage, int KvHeads>
__device__ __forceinline__ void prepare_kv(const PrepareArguments& a, int head, int t, int b,
                                           int lane, float2 sc, float* scratch = nullptr) {
    const auto m          = t + std::int64_t(b) * a.tokens;
    const int p           = a.positions[m];
    const auto page       = a.tables[p / 64 + std::int64_t(a.table_rows[b]) * a.table_stride];
    auto* key             = a.keys.at(p % 64, head, page);
    auto* value           = a.values.at(p % 64, head, page);
    const auto* raw_key   = a.k + (m * KvHeads + head) * 256;
    const auto* raw_value = a.v + (m * KvHeads + head) * 256;
    if constexpr (Storage == KvCacheStorage::BFloat16) {
        prepare_head<256>(raw_key, a.wk, reinterpret_cast<__nv_bfloat16*>(key), a.epsilon, sc,
                          lane);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            reinterpret_cast<__half*>(value)[lane + 32 * j] =
                __float2half_rn(__bfloat162float(raw_value[lane + 32 * j]));
    } else {
        float values[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) values[j] = __bfloat162float(raw_key[lane + 32 * j]);
        norm_rotate<256>(values, a.wk, a.epsilon, sc, lane);
        // The transformed FP32 key enters the codec directly, without a BF16 intermediate.
        kv::normalized_hadamard_d256_inplace(values, lane);
        constexpr auto key_storage   = kCachePlaneFormat<Storage, true>;
        constexpr auto value_storage = kCachePlaneFormat<Storage, false>;
        store_quantized<key_storage>(values, key, a.key_scales.at(p % 64, head, page), lane,
                                     scratch);
#pragma unroll
        for (int j = 0; j < 8; ++j) values[j] = __bfloat162float(raw_value[lane + 32 * j]);
        if constexpr (value_storage == KvCacheStorage::Nvfp4Group16)
            kv::normalized_hadamard_d256_inplace(values, lane);
        store_quantized<value_storage>(values, value, a.value_scales.at(p % 64, head, page), lane,
                                       scratch);
    }
}

__device__ __forceinline__ void store_fp8_contiguous(const float (&values)[8], std::uint8_t* codes,
                                                     std::uint8_t* scales, int lane) {
    float maximum = 0;
#pragma unroll
    for (float x : values) maximum = fmaxf(maximum, fabsf(x));
    const auto quant = common::fp8_fp16_scale(common::warp_max(maximum));
    unsigned words[2]{};
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const auto bits =
            quant.inverse_scale > 0
                ? __nv_cvt_float2_to_fp8x2(make_float2(values[2 * pair] * quant.inverse_scale,
                                                       values[2 * pair + 1] * quant.inverse_scale),
                                           __NV_SATFINITE, __NV_E4M3)
                : 0;
        words[pair / 2] |= unsigned(bits) << (16 * (pair % 2));
    }
    auto* output = codes + 8 * lane;
    if (reinterpret_cast<std::uintptr_t>(output) % 8 == 0)
        *reinterpret_cast<uint2*>(output) = make_uint2(words[0], words[1]);
    else {
#pragma unroll
        for (int j = 0; j < 8; ++j) output[j] = std::uint8_t(words[j / 4] >> (8 * (j % 4)));
    }
    if (lane == 0) *reinterpret_cast<__half*>(scales) = quant.scale;
}

__device__ __forceinline__ void store_int8_contiguous(const float (&values)[8], std::uint8_t* codes,
                                                      std::uint8_t* scales, int lane) {
    float maximum = 0;
#pragma unroll
    for (float x : values) maximum = fmaxf(maximum, fabsf(x));
    // Eight contiguous lanes cover one G64 group, using its represented FP16 scale.
    const auto quant = common::int8_fp16_scale(common::warp_max<8>(maximum));
    unsigned words[2]{};
#pragma unroll
    for (int j = 0; j < 8; ++j)
        words[j / 4] |=
            unsigned(std::uint8_t(common::int8_quant_code(values[j], quant.inverse_scale)))
            << (8 * (j % 4));
    auto* output = codes + 8 * lane;
    if (reinterpret_cast<std::uintptr_t>(output) % 8 == 0)
        *reinterpret_cast<uint2*>(output) = make_uint2(words[0], words[1]);
    else {
#pragma unroll
        for (int j = 0; j < 8; ++j) output[j] = std::uint8_t(words[j / 4] >> (8 * (j % 4)));
    }
    if (lane % 8 == 0) reinterpret_cast<__half*>(scales)[lane / 8] = quant.scale;
}

__device__ __forceinline__ void store_nvfp4_contiguous(const float (&values)[8],
                                                       std::uint8_t* codes, std::uint8_t* scales,
                                                       int lane) {
    float maximum = 0;
#pragma unroll
    for (float x : values) maximum = fmaxf(maximum, fabsf(x));
    // Two lanes own a G16 group; the shared codec owns its represented scale boundary.
    const auto scale = common::nvfp4_fp32_scale(common::warp_max<2>(maximum));
    unsigned word    = 0;
    if (scale.code) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 pair = make_float2(__fdiv_rn(values[2 * i], scale.represented),
                                            __fdiv_rn(values[2 * i + 1], scale.represented));
            word |= unsigned(__nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest))
                    << (8 * i);
        }
    }
    auto* output = codes + lane * 4;
    if (reinterpret_cast<std::uintptr_t>(output) % 4 == 0)
        *reinterpret_cast<unsigned*>(output) = word;
    else {
#pragma unroll
        for (int i = 0; i < 4; ++i) output[i] = std::uint8_t(word >> (8 * i));
    }
    if (lane % 2 == 0) scales[lane / 2] = scale.code;
}

template <KvCacheStorage Storage>
__device__ __forceinline__ void store_quantized_contiguous(const float (&values)[8],
                                                           std::uint8_t* codes,
                                                           std::uint8_t* scales, int lane) {
    if constexpr (Storage == KvCacheStorage::Int8Group64)
        store_int8_contiguous(values, codes, scales, lane);
    else if constexpr (Storage == KvCacheStorage::Nvfp4Group16)
        store_nvfp4_contiguous(values, codes, scales, lane);
    else {
        static_assert(Storage == KvCacheStorage::Fp8E4M3Row256);
        store_fp8_contiguous(values, codes, scales, lane);
    }
}

// Natural cache alignment is two bytes, while the common packed row admits a 16-byte store.
template <bool Half>
__device__ __forceinline__ void store_16bit_contiguous(const float (&values)[8],
                                                       std::uint8_t* output, int lane) {
    unsigned words[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if constexpr (Half)
            words[j] = common::pack_f16x2(values[2 * j], values[2 * j + 1]);
        else
            words[j] = common::pack_bf16x2(values[2 * j], values[2 * j + 1]);
    }
    if (reinterpret_cast<std::uintptr_t>(output) % 16 == 0)
        reinterpret_cast<uint4*>(output)[lane] = {words[0], words[1], words[2], words[3]};
    else {
#pragma unroll
        for (int j = 0; j < 8; ++j)
            reinterpret_cast<std::uint16_t*>(output)[lane * 8 + j] =
                std::uint16_t(words[j / 2] >> (16 * (j % 2)));
    }
}

// A warp consumes eight adjacent features per lane. Each codec receives the
// norm/RoPE result at its declared rounding boundary, without an extra K cast.
template <KvCacheStorage Storage, int KvHeads>
__device__ __forceinline__ void prepare_kv_vector(const PrepareArguments& a, int head, int t, int b,
                                                  int lane, const float2* coefficients) {
    const auto m      = t + std::int64_t(b) * a.tokens;
    const int p       = a.positions[m];
    const auto page   = a.tables[p / 64 + std::int64_t(a.table_rows[b]) * a.table_stride];
    const auto* key   = a.k + (m * KvHeads + head) * 256;
    const auto* value = a.v + (m * KvHeads + head) * 256;
    float x[8];
    if (reinterpret_cast<std::uintptr_t>(a.wk) % 16 == 0)
        vector_norm_rotate<256, true>(key, a.wk, a.epsilon, coefficients, lane, x);
    else
        vector_norm_rotate<256, false>(key, a.wk, a.epsilon, coefficients, lane, x);
    if constexpr (Storage == KvCacheStorage::BFloat16) {
        store_16bit_contiguous<false>(x, a.keys.at(p % 64, head, page), lane);
    } else {
        kv::normalized_hadamard_d256_contiguous_inplace(x, lane);
        store_quantized_contiguous<kCachePlaneFormat<Storage, true>>(
            x, a.keys.at(p % 64, head, page), a.key_scales.at(p % 64, head, page), lane);
    }
    const auto v = reinterpret_cast<const uint4*>(value)[lane];
    const unsigned words[4]{v.x, v.y, v.z, v.w};
#pragma unroll
    for (int j = 0; j < 8; ++j)
        x[j] = __uint_as_float(((words[j / 2] >> (16 * (j % 2))) & 0xffff) << 16);
    if constexpr (Storage == KvCacheStorage::BFloat16)
        store_16bit_contiguous<true>(x, a.values.at(p % 64, head, page), lane);
    else {
        if constexpr (kCachePlaneFormat<Storage, false> == KvCacheStorage::Nvfp4Group16)
            kv::normalized_hadamard_d256_contiguous_inplace(x, lane);
        store_quantized_contiguous<kCachePlaneFormat<Storage, false>>(
            x, a.values.at(p % 64, head, page), a.value_scales.at(p % 64, head, page), lane);
    }
}

} // namespace ninfer::ops::detail::qsa
