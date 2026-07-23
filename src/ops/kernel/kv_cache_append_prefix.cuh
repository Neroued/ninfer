#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheAppendPrefixHeadDim = 128;
inline constexpr int kKVCacheAppendPrefixHeads   = 8;
inline constexpr int kKVCacheAppendPrefixWindow  = 4096;

template <int VectorBytes>
__device__ __forceinline__ void kv_cache_append_prefix_copy_vector(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, int token,
    int unit_in_token, int slot, int padded_capacity) {
    static_assert(VectorBytes == 16 || VectorBytes == 32);
    constexpr int Bf16PerVector  = VectorBytes / static_cast<int>(sizeof(__nv_bfloat16));
    constexpr int VectorsPerHead = kKVCacheAppendPrefixHeadDim / Bf16PerVector;
    const int kv_head            = unit_in_token / VectorsPerHead;
    const int d                  = (unit_in_token - kv_head * VectorsPerHead) * Bf16PerVector;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst = static_cast<std::int64_t>(d) +
                             static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                 (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);

    const int4 k0                           = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0                           = *reinterpret_cast<const int4*>(&v[src]);
    *reinterpret_cast<int4*>(&cache_k[dst]) = k0;
    *reinterpret_cast<int4*>(&cache_v[dst]) = v0;
    if constexpr (VectorBytes == 32) {
        const int4 k1                               = *reinterpret_cast<const int4*>(&k[src + 8]);
        const int4 v1                               = *reinterpret_cast<const int4*>(&v[src + 8]);
        *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
        *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
    }
}

template <bool Cyclic, int VectorBytes, bool Persistent>
__global__ void kv_cache_append_prefix_flat_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ commit_count,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, int min_count,
    int max_count, int padded_capacity) {
    constexpr int Bf16PerVector = VectorBytes / static_cast<int>(sizeof(__nv_bfloat16));
    constexpr int UnitsPerToken =
        kKVCacheAppendPrefixHeads * kKVCacheAppendPrefixHeadDim / Bf16PerVector;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256);
    const int count = commit_count[0];
    if (count < min_count || count > max_count) return;

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    if constexpr (Persistent) {
        const int groups = max_count == 0 ? 0 : 1 + (max_count - 1) / TokensPerBlock;
        for (int group = 0; group < groups; ++group) {
            const int token = group * TokensPerBlock + local_token;
            if (token >= count) continue;
            const int position = positions[token];
            const int slot     = Cyclic ? position & (kKVCacheAppendPrefixWindow - 1) : position;
            kv_cache_append_prefix_copy_vector<VectorBytes>(k, v, cache_k, cache_v, token,
                                                            unit_in_token, slot, padded_capacity);
        }
    } else {
        const int token = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
        if (token >= count) return;
        const int position = positions[token];
        const int slot     = Cyclic ? position & (kKVCacheAppendPrefixWindow - 1) : position;
        kv_cache_append_prefix_copy_vector<VectorBytes>(k, v, cache_k, cache_v, token,
                                                        unit_in_token, slot, padded_capacity);
    }
}

template <bool Cyclic>
__global__ void kv_cache_append_prefix_token_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ commit_count,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, int min_count,
    int max_count, int padded_capacity) {
    const int count = commit_count[0];
    const int token = static_cast<int>(blockIdx.x);
    if (count < min_count || count > max_count || token >= count) return;
    const int position = positions[token];
    const int slot     = Cyclic ? position & (kKVCacheAppendPrefixWindow - 1) : position;
    kv_cache_append_prefix_copy_vector<16>(k, v, cache_k, cache_v, token,
                                           static_cast<int>(threadIdx.x), slot, padded_capacity);
}

} // namespace ninfer::ops
