#pragma once

#include "ops/softmax_attention/kernels/causal/tile_io.cuh"
#include "ops/kv_cache/kernels/hadamard.cuh"

namespace ninfer::ops::detail {

__device__ __forceinline__ void causal_store_partial_pair(float* target, float a, float b) {
    *reinterpret_cast<float2*>(target) = make_float2(a, b);
}

template <class G>
__device__ __forceinline__ void causal_store_output_pair(__nv_bfloat16* out, int head, int d,
                                                         int token, float a, float b) {
    *reinterpret_cast<unsigned*>(out + causal_q_index<G>(head, d, token)) =
        common::pack_bf16x2(a, b);
}

__device__ __forceinline__ void causal_store_output(__nv_bfloat16* out, float value) {
    *out = __float2bfloat16(value);
}

// A warp converts one normalized row from the represented rotated V coordinates.
template <class G>
__device__ __forceinline__ void
causal_store_inverse_rotated_row(const float* row, __nv_bfloat16* out, int head, int token) {
    static_assert(G::kHeadDim == kCausalHeadDim);
    const int lane = threadIdx.x & 31;
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) values[r] = row[lane + 32 * r];
    kv::normalized_hadamard_d256_inplace(values, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r)
        causal_store_output(out + causal_q_index<G>(head, lane + 32 * r, token), values[r]);
}

} // namespace ninfer::ops::detail
