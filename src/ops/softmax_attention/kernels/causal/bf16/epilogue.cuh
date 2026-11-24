#pragma once

#include "ops/softmax_attention/kernels/causal/bf16/tile_io.cuh"

namespace ninfer::ops::detail {

// Both mainloops publish either FP32 (unnormalized numerator, log2 maximum, sum)
// or one final BF16 result. No low-precision partial materialization is involved.
template <class G, bool Partial>
__device__ __forceinline__ void bf16_kv_store_pair(CausalPartialView partial, __nv_bfloat16* out,
                                                   int head, int token, int d, int width, int split,
                                                   float a, float b, float maximum, float sum) {
    if constexpr (Partial) {
        const auto offset = causal_partial_index<G>(head, d, token, split, width);
        *reinterpret_cast<float2*>(partial.acc + offset) = make_float2(a, b);
        if (d == 0) {
            const auto stat       = causal_stat_index<G>(head, token, split, width);
            partial.maximum[stat] = maximum;
            partial.sum[stat]     = sum;
        }
    } else {
        const float inv = sum > 0.0f ? __frcp_rn(sum) : 0.0f;
        *reinterpret_cast<unsigned*>(out + causal_q_index<G>(head, d, token)) =
            common::pack_bf16x2(a * inv, b * inv);
    }
}

} // namespace ninfer::ops::detail
