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

template <class G, int Threads>
__device__ __forceinline__ void causal_zero_grouped_output(__nv_bfloat16* out, int head,
                                                           std::int64_t column, int tokens) {
    for (int i = threadIdx.x; i < tokens * G::GroupSize * G::kHeadDim; i += Threads) {
        const int row = i / G::kHeadDim, d = i % G::kHeadDim;
        out[causal_q_index<G>(head * G::GroupSize + row % G::GroupSize, d,
                              static_cast<int>(column) + row / G::GroupSize)] =
            __float2bfloat16(0.F);
    }
}

template <class G>
__device__ __forceinline__ void causal_store_ragged_output_pair(
    __nv_bfloat16* out, int head, int d, int token, float a, float b) {
    auto* dst = out + causal_q_index<G>(head, d, token);
    if ((reinterpret_cast<std::uintptr_t>(dst) & 3U) == 0U) {
        *reinterpret_cast<unsigned*>(dst) = common::pack_bf16x2(a, b);
    } else {
        dst[0] = __float2bfloat16(a);
        dst[1] = __float2bfloat16(b);
    }
}

#define NINFER_CAUSAL_EPILOGUE causal_grouped_epilogue
#define NINFER_CAUSAL_OUTPUT_PAIR causal_store_output_pair
#include "ops/softmax_attention/kernels/causal/grouped_epilogue.inl"
#undef NINFER_CAUSAL_OUTPUT_PAIR
#undef NINFER_CAUSAL_EPILOGUE

#define NINFER_CAUSAL_EPILOGUE causal_ragged_grouped_epilogue
#define NINFER_CAUSAL_OUTPUT_PAIR causal_store_ragged_output_pair
#include "ops/softmax_attention/kernels/causal/grouped_epilogue.inl"
#undef NINFER_CAUSAL_OUTPUT_PAIR
#undef NINFER_CAUSAL_EPILOGUE

} // namespace ninfer::ops::detail
