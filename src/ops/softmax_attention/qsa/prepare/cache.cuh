#pragma once

#include "ninfer/types.h"
#include "ops/kv_cache/kernels/fp8.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/common/quant/integer.cuh"
#include "ops/common/quant/nvfp4.cuh"
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
    kv::store_fp8_d256(values, codes, scales, lane);
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
    else if constexpr (Storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0)
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
        constexpr auto key_storage = Storage == KvCacheStorage::Fp8KeyNvfp4Value
                                         ? KvCacheStorage::Fp8E4m3fnGroup128Ue8m0
                                         : Storage;
        constexpr auto value_storage =
            Storage == KvCacheStorage::Fp8KeyNvfp4Value ? KvCacheStorage::Nvfp4Group16 : Storage;
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

} // namespace ninfer::ops::detail::qsa
