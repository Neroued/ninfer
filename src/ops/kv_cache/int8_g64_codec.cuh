#pragma once

// Signed int8, per-token G64 KV-cache codec shared by append and causal-attention kernels.
// This header owns index math, vectorized decode, and scalar encode; there is deliberately no
// standalone transcode kernel in the production path.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheInt8HeadDim = 256;
inline constexpr int kKVCacheInt8Group   = 64;
inline constexpr int kKVCacheInt8Groups  = kKVCacheInt8HeadDim / kKVCacheInt8Group;

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_code_index(int physical_page, int kv_head, int d, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8HeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                           page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t
kv_cache_int8_quant_scale_index(int physical_page, int kv_head, int group, int page_offset) {
    return paged_kv_element_offset<kKVCacheInt8Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                          page_offset, group);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t kv_cache_int8_quant_src_index(int kv_head, int d,
                                                                      int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kKVCacheInt8HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

struct KVCacheInt8QuantParams {
    __half scale;
    float inverse_scale;
};

// Exact persistent group-scale boundary shared by standalone and fused append. The stored scale is
// FP16-RNE(absmax/127); codes always use the reciprocal of that represented FP16 value.
__device__ __forceinline__ KVCacheInt8QuantParams kv_cache_int8_quant_params(float absmax) {
    const __half scale            = __float2half_rn(absmax > 0.0f ? absmax / 127.0f : 0.0f);
    const float represented_scale = __half2float(scale);
    return {
        .scale         = scale,
        .inverse_scale = represented_scale > 0.0f ? 1.0f / represented_scale : 0.0f,
    };
}

__device__ __forceinline__ std::int8_t kv_cache_int8_quant_code(float x, float inv_scale) {
    if (inv_scale == 0.0f) { return static_cast<std::int8_t>(0); }
    int q = __float2int_rn(x * inv_scale);
    q     = max(-127, min(127, q));
    return static_cast<std::int8_t>(q);
}

__device__ __forceinline__ int4 kv_cache_int8_dequant_i8x8_from(const std::int8_t* codes8,
                                                                float s) {
    const int2 raw       = load_vec<int2>(codes8);
    const std::int8_t* c = reinterpret_cast<const std::int8_t*>(&raw);
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float x0 = static_cast<float>(c[2 * i]) * s;
        const float x1 = static_cast<float>(c[2 * i + 1]) * s;
        packed[i]      = pack_bf16x2(x0, x1);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

} // namespace ninfer::ops
