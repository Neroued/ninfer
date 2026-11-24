#pragma once

#include "core/tensor.h"
#include "ops/common/numeric/math.h"
#include "ops/linear_attention/kimi_delta_attention/operands.h"

namespace ninfer::ops::detail::kimi_delta_attention {
inline constexpr int kChunkSize = 16;

__host__ __device__ inline int chunk_count(int tokens) {
    return common::div_up(tokens, kChunkSize);
}

// One producer/consumer packet. The vector and square tiles use the swizzles below in both
// global and shared memory, so recurrence can asynchronously copy whole packets without repacking.
struct alignas(256) Chunk {
    __nv_bfloat16 kd[kChunkSize * kStateDim];
    __nv_bfloat16 qd[kChunkSize * kStateDim];
    __nv_bfloat16 kr[kChunkSize * kStateDim];
    float gamma[kStateDim];
    float solve[kChunkSize * kChunkSize];
    float mqk[kChunkSize * kChunkSize];
};

static_assert(sizeof(Chunk) == 14848);

__host__ __device__ constexpr int vector_index(int row, int col) {
    return row * kStateDim + (col ^ (row * 8));
}

// Kr time rows are interleaved {0,4,1,5,2,6,3,7} for transposed BF16 MMA loads.
__host__ __device__ constexpr int restored_index(int row, int col) {
    const int physical = (row & ~7) + (row & 3) * 2 + ((row >> 2) & 1);
    return vector_index(physical, col);
}

__host__ __device__ constexpr int square_index(int row, int col) {
    return row * kChunkSize + (col ^ (((row >> 1) & 3) * 4));
}

inline std::size_t chunked_workspace_bytes(int value_heads, int tokens) {
    return Tensor::metadata(DType::U8,
                            {static_cast<int>(sizeof(Chunk)), chunk_count(tokens), value_heads})
        .bytes();
}

} // namespace ninfer::ops::detail::kimi_delta_attention
