#pragma once

// Asymmetric K8V4 append kernel. K uses the existing row-256 E4M3 cache codec; V uses the
// group-16 packed E2M1 cache codec. Both operands receive the fixed FP32 D256 Hadamard rotation.

#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kv_cache/append/geometry.cuh"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/nvfp4_group16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry>
__device__ __forceinline__ void kv_cache_append_full_k8v4_row(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, int token, int kv_head,
    int physical_page, int page_offset, int lane, float* scratch) {
    constexpr unsigned FullMask = 0xffffffffU;
    float values[8];
    float local_absmax = 0.0F;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(k[kv_cache_fp8_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (float value : values) local_absmax = fmaxf(local_absmax, fabsf(value));
    const auto k_quant = kv_cache_fp8_quant_params(warp_max(local_absmax, FullMask));
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        cache_k[kv_cache_fp8_code_index<Geometry>(physical_page, kv_head, d, page_offset)] =
            kv_cache_fp8_quant_code(values[r], k_quant.inverse_scale);
    }
    if (lane == 0) {
        scale_k[kv_cache_fp8_scale_index<Geometry>(physical_page, kv_head, page_offset)] =
            k_quant.scale;
    }

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(v[kv_cache_nvfp4_src_index<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) scratch[lane + 32 * r] = values[r];
    __syncwarp();
    if (lane < kKVCacheNvfp4Groups) {
        const auto quantized = kv_cache_nvfp4_quantize_group16(scratch + lane * kKVCacheNvfp4Group);
        const std::int64_t code_offset = kv_cache_nvfp4_code_index<Geometry>(
            physical_page, kv_head, lane * kKVCacheNvfp4Group, page_offset);
        store_vec(cache_v + code_offset, make_uint2(quantized.codes_lo, quantized.codes_hi));
        scale_v[kv_cache_nvfp4_scale_index<Geometry>(physical_page, kv_head, lane, page_offset)] =
            quantized.scale;
    }
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void kv_cache_append_full_k8v4_kernel(const __nv_bfloat16* __restrict__ k,
                                          const __nv_bfloat16* __restrict__ v,
                                          const std::int32_t* __restrict__ positions,
                                          Metadata metadata, std::uint8_t* __restrict__ cache_k,
                                          std::uint8_t* __restrict__ cache_v,
                                          __half* __restrict__ scale_k,
                                          std::uint8_t* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    __shared__ float scratch[Warps][kKVCacheNvfp4HeadDim];
    const int tokens = metadata.valid_tokens(width);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int unit   = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units  = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    physical_page                   = __shfl_sync(FullMask, physical_page, 0);
    kv_cache_append_full_k8v4_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token,
                                            kv_head, physical_page, position & kPagedKVPageMask,
                                            lane, scratch[warp]);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void kv_cache_append_full_k8v4_page_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    __half* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    __shared__ float scratch[Warps][kKVCacheNvfp4HeadDim];
    const int tokens        = metadata.valid_tokens(width);
    const int warp          = static_cast<int>(threadIdx.x) >> 5;
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int kv_head       = static_cast<int>(blockIdx.y);
    const int base_position = positions[0];
    const int tile_position =
        (base_position / TokensPerTile + static_cast<int>(blockIdx.x)) * TokensPerTile;
    const int token_begin = max(0, tile_position - base_position);
    const int token_end   = min(tokens, tile_position + TokensPerTile - base_position);
    const int token       = token_begin + warp;
    if (token >= token_end) return;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page  = lane == 0 ? block_table[tile_position >> kPagedKVPageShift] : 0;
    physical_page      = __shfl_sync(FullMask, physical_page, 0);
    const int position = base_position + token;
    kv_cache_append_full_k8v4_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token,
                                            kv_head, physical_page, position & kPagedKVPageMask,
                                            lane, scratch[warp]);
}

} // namespace ninfer::ops
