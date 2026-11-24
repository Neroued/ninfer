#pragma once
#include "ops/common/numeric/math.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/softmax_attention/kernels/causal/geometry.h"
#include <cuda_bf16.h>

namespace ninfer::ops::detail {
template <class G>
__device__ __forceinline__ std::int64_t causal_q_index(int head, int d, int token = 0) {
    return d + static_cast<std::int64_t>(G::kHeadDim) *
                   (head + static_cast<std::int64_t>(G::QHeads) * token);
}

template <class G>
__device__ __forceinline__ std::int64_t causal_new_index(int head, int d, int token = 0) {
    return d + static_cast<std::int64_t>(G::kHeadDim) *
                   (head + static_cast<std::int64_t>(G::KVHeads) * token);
}

template <class G>
__device__ __forceinline__ std::int64_t causal_partial_index(int head, int d, int token, int split,
                                                             int tokens) {
    return d + static_cast<std::int64_t>(G::kHeadDim) *
                   (head + static_cast<std::int64_t>(G::QHeads) *
                               (token + static_cast<std::int64_t>(tokens) * split));
}

template <class G>
__device__ __forceinline__ std::int64_t causal_stat_index(int head, int token, int split,
                                                          int tokens) {
    return head + static_cast<std::int64_t>(G::QHeads) *
                      (token + static_cast<std::int64_t>(tokens) * split);
}

// Keep the explicit empty guard and row/d form: the compact modulo loop
// caused register spills in sm_120a prefill kernels.
template <typename Geometry>
__device__ __forceinline__ void causal_zero_rows(__nv_bfloat16* out, int q_head, int row_begin,
                                                 int row_end, int tid, int threads) {
    if (row_begin >= row_end) { return; }
    const int elements = (row_end - row_begin) * Geometry::kHeadDim;
    for (int element = tid; element < elements; element += threads) {
        const int row = row_begin + element / Geometry::kHeadDim;
        const int d   = element - (row - row_begin) * Geometry::kHeadDim;
        out[causal_q_index<Geometry>(q_head, d, row)] = __float2bfloat16(0.0f);
    }
}

template <typename Byte>
__device__ __forceinline__ void causal_store_query_code(Byte* tile, int row, int d, Byte code) {
    const int col_b16 = d >> 1;
    const int byte    = d & 1;
    const int off = (row * (kCausalHeadDim / 2) + common::swizzle_xor_64(row, col_b16)) * 2 + byte;
    tile[off]     = code;
}

template <typename Geometry>
__device__ __forceinline__ void causal_row_to_qt(int row, int kv_head, int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

template <int Columns>
__device__ __forceinline__ int causal_probability_swizzle(int row, int col) {
    if constexpr (Columns == 32) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
    return common::swizzle_xor_64(row, col);
}

} // namespace ninfer::ops::detail
