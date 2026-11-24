#pragma once

#include "ops/softmax_attention/kernels/causal/ragged_grouped.cuh"

#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/epilogue.cuh"
#include "ops/softmax_attention/kernels/softmax.cuh"
#include "ops/softmax_attention/kernels/causal/bf16/partition.h"
#include "ops/softmax_attention/kernels/causal/partition.h"

namespace ninfer::ops::detail {

template <class G, class S, class Input>
__device__ __forceinline__ void
bf16_kv_load_grouped_tile(__nv_bfloat16* key_tile, __half* value_tile, const __nv_bfloat16* keys,
                          const __half* values, Input input, std::int64_t page_offset, int head,
                          int begin, int end, int first_position, int tid) {
    constexpr int D       = G::kHeadDim;
    const auto cache_base = bf16_kv_cache_index<G>(page_offset, head, 0, begin & kv::kPageMask);
    // Small query tiles need enough outstanding loads to hide memory latency.
#pragma unroll 1
    for (int i = tid; i < S::kKeyRows * (D / 8); i += S::kThreads) {
        const int row = i / (D / 8), d = i % (D / 8) * 8, key = begin + row;
        const auto offset = row * D + common::swizzle_xor_64(row, d);
        auto* kd          = key_tile + offset;
        auto* vd          = value_tile + offset;
        if (key >= end) {
            common::store_vec(kd, make_int4(0, 0, 0, 0));
            common::store_vec(vd, make_int4(0, 0, 0, 0));
        } else {
            if constexpr (Input::writes_cache) {
                if (key >= first_position) {
                    const auto index = causal_new_index<G>(head, d, key - first_position);
                    common::cp_async<16>(kd, input.k + index);
                    common::store_vec(vd, common::bf16x8_bits_to_f16x8_bits(
                                              common::load_vec<int4>(input.v + index)));
                    continue;
                }
            }
            const auto index = cache_base + row * D + d;
            common::cp_async<16>(kd, keys + index);
            common::cp_async<16>(vd, values + index);
        }
    }
}

// Packed-query tiles and KV partitions are independent grid axes. Each KV row
// has one append owner; all query CTAs read new rows from the immutable inputs.
#include "ops/softmax_attention/kernels/causal/bf16/grouped_mma.inl"

#define NINFER_CAUSAL_RAGGED
#include "ops/softmax_attention/kernels/causal/bf16/grouped_mma.inl"
#undef NINFER_CAUSAL_RAGGED

} // namespace ninfer::ops::detail
