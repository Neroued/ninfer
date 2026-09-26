#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int IntermediateRows>
struct Q8SwiGluPairedRows {
    static_assert(IntermediateRows > 0 && (IntermediateRows % 8) == 0);
    static constexpr int kOutputRowsPerCta = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? IntermediateRows : 0);
    }
};

// The admitted SwiGLU geometries contain complete eight-row output tiles.
struct Q8SwiGluDirectEpilogue {
    template <class Output>
    __device__ __forceinline__ void store_fragment(const Output& output, int row, int col,
                                                   float4 projected, int /*rows*/,
                                                   int columns) const {
        if (col < columns) output.store(row, col, silu(projected.x) * projected.z);
        if (col + 1 < columns) output.store(row, col + 1, silu(projected.y) * projected.w);
    }
};
} // namespace ninfer::ops::detail
