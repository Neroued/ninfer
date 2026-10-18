#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/kv_cache/kernels/layout.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail::kv {
template <typename Geometry, typename Metadata, bool MultiBatch = false>
__global__ void write_bf16_kernel(const __nv_bfloat16* __restrict__ k,
                                  const __nv_bfloat16* __restrict__ v,
                                  const std::int32_t* __restrict__ positions, Metadata metadata,
                                  __nv_bfloat16* __restrict__ cache_k, __half* __restrict__ cache_v,
                                  std::int32_t width) {
    if constexpr (MultiBatch) {
        const int batch = blockIdx.z;
        metadata.table_rows += batch;
        if (metadata.valid_columns) metadata.valid_columns += batch;
        const auto offset = static_cast<std::int64_t>(batch) * width * kHeadDim * Geometry::KVHeads;
        k += offset;
        v += offset;
        positions += static_cast<std::int64_t>(batch) * width;
    }
    constexpr int VecElems = 8;
    const int tokens       = metadata.valid_tokens(width);
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n =
        static_cast<std::int64_t>(tokens) * Geometry::KVHeads * (kHeadDim / VecElems);
    if (idx >= n) return;

    const int vec                   = static_cast<int>(idx % (kHeadDim / VecElems));
    const int tmp                   = static_cast<int>(idx / (kHeadDim / VecElems));
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int d                     = vec * VecElems;
    const int position              = positions[0] + token;
    const int lane                  = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? lookup_page(block_table, position) : 0;
    const std::int64_t src_off =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(kHeadDim) * (kv_head + Geometry::KVHeads * token);
    const int4 k_value = common::load_vec<int4>(&k[src_off]);
    const int4 v_value = common::bf16x8_bits_to_f16x8_bits(common::load_vec<int4>(&v[src_off]));
    physical_page      = __shfl_sync(0xffffffffu, physical_page, 0);
    const std::int64_t cache_off = page_major_offset<kHeadDim, Geometry::KVHeads>(
        physical_page, kv_head, position & kPageMask, d);
    common::store_vec(&cache_k[cache_off], k_value);
    common::store_vec(&cache_v[cache_off], v_value);
}

} // namespace ninfer::ops::detail::kv
