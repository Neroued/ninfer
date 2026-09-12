#pragma once

// E4M3FN row-scaled D256 KV-cache codec shared by standalone append and causal Attention.
// Persistent scales cross one FP16 represented-value boundary before codes are formed.

#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheFp8HeadDim        = 256;
inline constexpr int kKVCacheFp8Group          = 256;
inline constexpr int kKVCacheFp8Groups         = 1;
inline constexpr float kKVCacheFp8MaxFinite    = 448.0F;
inline constexpr float kKVCacheFp8ScaleMinimum = 0x1p-24F;
inline constexpr float kKVCacheFp8ScaleMaximum = 65504.0F;

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_fp8_code_index(int physical_page, int kv_head,
                                                                int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheFp8HeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_fp8_scale_index(int physical_page, int kv_head,
                                                                 int page_offset) {
    return paged_kv_element_offset<kKVCacheFp8Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                         page_offset, 0);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_fp8_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheFp8HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheFp8QuantParams {
    __half scale;
    float inverse_scale;
};

__device__ __forceinline__ KVCacheFp8QuantParams kv_cache_fp8_quant_params(float absmax) {
    if (absmax == 0.0F) { return {.scale = __float2half_rn(0.0F), .inverse_scale = 0.0F}; }
    const float raw_scale = absmax / kKVCacheFp8MaxFinite;
    const float bounded = fminf(kKVCacheFp8ScaleMaximum, fmaxf(kKVCacheFp8ScaleMinimum, raw_scale));
    const __half scale  = __float2half_rn(bounded);
    const float represented_scale = __half2float(scale);
    return {.scale = scale, .inverse_scale = 1.0F / represented_scale};
}

__device__ __forceinline__ std::uint8_t kv_cache_fp8_quant_code(float x, float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    return __nv_cvt_float_to_fp8(x * inverse_scale, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ std::uint16_t kv_cache_fp8_quant_code2(float x0, float x1,
                                                                  float inverse_scale) {
    if (inverse_scale == 0.0F) { return 0; }
    return __nv_cvt_float2_to_fp8x2(make_float2(x0 * inverse_scale, x1 * inverse_scale),
                                    __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ __half2 kv_cache_fp8_code2_to_half2(std::uint16_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = storage;
    return static_cast<__half2>(value);
}

__device__ __forceinline__ __half2 kv_cache_fp8_dequant_code2_to_half2(std::uint16_t storage,
                                                                       __half scale) {
    return __hmul2(kv_cache_fp8_code2_to_half2(storage), __halves2half2(scale, scale));
}

} // namespace ninfer::ops
