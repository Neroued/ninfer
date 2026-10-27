#pragma once

// Asymmetric K8V4 append kernel. K uses the existing G128 E4M3/UE8M0 cache codec; V uses the
// group-16 packed E2M1 cache codec. Both operands receive the fixed FP32 D256 Hadamard rotation.

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/kv_cache/kernels/fp8.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/common/quant/nvfp4.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail::kv {

template <typename Geometry>
__device__ __forceinline__ void
write_k8v4_row(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
               std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
               std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, int token,
               int kv_head, std::int64_t record, int page_offset, int lane, float* scratch) {
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(k[source_offset<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
    store_fp8_d256(values, cache_k + Fp8G128Layout::code_offset(record, kv_head, 0, page_offset),
                   scale_k + Fp8G128Layout::scale_offset(record, kv_head, page_offset), lane);

#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int d = lane + 32 * r;
        values[r]   = __bfloat162float(v[source_offset<Geometry>(kv_head, d, token)]);
    }
    normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) scratch[lane + 32 * r] = values[r];
    __syncwarp();
    if (lane < kNvfp4Groups) {
        const auto quantized = common::quantize_nvfp4_group16_f32(scratch + lane * kNvfp4Group);
        const std::int64_t code_offset =
            Nvfp4G16Layout::code_offset(record, kv_head, lane * kNvfp4Group, page_offset);
        common::store_vec(cache_v + code_offset,
                          make_uint2(quantized.codes_lo, quantized.codes_hi));
        scale_v[Nvfp4G16Layout::scale_offset(record, kv_head, lane, page_offset)] = quantized.scale;
    }
}

template <typename Geometry, typename Metadata, bool MultiBatch = false>
__launch_bounds__(256) __global__
    void write_k8v4_kernel(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                           const std::int32_t* __restrict__ positions, Metadata metadata,
                           std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                           std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v,
                           std::int32_t width) {
    if constexpr (MultiBatch) {
        const int batch = blockIdx.z;
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += batch * width;
    }
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    __shared__ float scratch[Warps][kHeadDim];
    const int tokens = metadata.valid_tokens(width);
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int unit   = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units  = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int64_t* block_table = metadata.block_table();
    std::int64_t record             = lane == 0 ? lookup_page(block_table, position) : 0;
    record                          = __shfl_sync(FullMask, record, 0);
    write_k8v4_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head, record,
                             position & kPageMask, lane, scratch[warp]);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void write_k8v4_page_kernel(const __nv_bfloat16* __restrict__ k,
                                const __nv_bfloat16* __restrict__ v,
                                const std::int32_t* __restrict__ positions, Metadata metadata,
                                std::uint8_t* __restrict__ cache_k,
                                std::uint8_t* __restrict__ cache_v,
                                std::uint8_t* __restrict__ scale_k,
                                std::uint8_t* __restrict__ scale_v, std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    __shared__ float scratch[Warps][kHeadDim];
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
    const std::int64_t* block_table = metadata.block_table();
    std::int64_t record             = lane == 0 ? block_table[tile_position >> kPageShift] : 0;
    record                          = __shfl_sync(FullMask, record, 0);
    const int position              = base_position + token;
    write_k8v4_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head, record,
                             position & kPageMask, lane, scratch[warp]);
}

} // namespace ninfer::ops::detail::kv
