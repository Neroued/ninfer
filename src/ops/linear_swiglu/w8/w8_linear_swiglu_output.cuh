#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int IntermediateRows>
struct W8SwiGluPairedRows {
    static_assert(IntermediateRows > 0 && (IntermediateRows % 8) == 0);
    static constexpr int kOutputRowsPerCta = 8;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + (local_row & 7) + (local_row >= 8 ? IntermediateRows : 0);
    }
};

struct W8SwiGluDirectEpilogue {
    __nv_bfloat16* out;
    std::int32_t rows;

    __device__ __forceinline__ void store_pair(int row, int col0, float4 projected,
                                               int columns) const {
        if (col0 < columns) {
            out[static_cast<std::int64_t>(col0) * rows + row] =
                __float2bfloat16_rn(silu(projected.x) * projected.z);
        }
        if (col0 + 1 < columns) {
            out[static_cast<std::int64_t>(col0 + 1) * rows + row] =
                __float2bfloat16_rn(silu(projected.y) * projected.w);
        }
    }
};

} // namespace ninfer::ops::detail
