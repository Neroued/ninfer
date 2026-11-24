#pragma once

#include "ops/softmax_attention/kernels/causal/ragged.h"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/softmax_attention/kernels/causal/nvfp4/schedule.cuh"
#include "ops/softmax_attention/kernels/causal/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"

namespace ninfer::ops::detail {
struct Nvfp4KvBarriers {
    alignas(8) std::uint64_t k_full;
    alignas(8) std::uint64_t k_empty;
    alignas(8) std::uint64_t v_full;
    alignas(8) std::uint64_t v_empty;
};

template <typename Geometry, typename Schedule>
__device__ __forceinline__ void
nvfp4_kv_decode_tile(__half* destination, const std::uint8_t* cache,
                     const std::uint8_t* cache_scale, const std::int64_t* block_table, int kv_head,
                     int tile_k0, int max_query_abs, int producer_tid) {
    constexpr int D                  = 256;
    constexpr int Bc                 = Schedule::kKeyRows;
    constexpr int GroupsPerTile      = Bc * kv::kNvfp4Groups;
    constexpr int ProducerThreads    = Schedule::kProducerThreads;
    const std::int64_t record_offset = block_table[tile_k0 >> kv::kPageShift];
    const int page_offset0           = tile_k0 & kv::kPageMask;

#pragma unroll 1
    for (int task = producer_tid; task < GroupsPerTile; task += ProducerThreads) {
        const int key_l   = task / kv::kNvfp4Groups;
        const int group   = task - key_l * kv::kNvfp4Groups;
        const int d       = group * kv::kNvfp4Group;
        const int key     = tile_k0 + key_l;
        __half* target_lo = destination + key_l * D + common::swizzle_xor_64(key_l, d);
        __half* target_hi = destination + key_l * D + common::swizzle_xor_64(key_l, d + 8);
        if (key <= max_query_abs) {
            const std::int64_t code_offset =
                kv::Nvfp4G16Layout::code_offset(record_offset, kv_head, d, page_offset0 + key_l);
            const std::int64_t scale_offset = kv::Nvfp4G16Layout::scale_offset(
                record_offset, kv_head, group, page_offset0 + key_l);
            const auto represented =
                common::nvfp4_dequant_f16x16(cache + code_offset, cache_scale[scale_offset]);
            common::store_vec(target_lo, represented.lo);
            common::store_vec(target_hi, represented.hi);
        } else {
            common::store_vec(target_lo, make_int4(0, 0, 0, 0));
            common::store_vec(target_hi, make_int4(0, 0, 0, 0));
        }
    }
}

#include "ops/softmax_attention/kernels/causal/nvfp4/tiled_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/nvfp4/tiled_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
