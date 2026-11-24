#pragma once

#include "ops/moe/kernels/cuda/output.cuh"
#include "ops/moe/kernels/cuda/block_scaled/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled {

__device__ __forceinline__ float routed_value(int token, int row, const Storage& storage,
                                              const float* coefficients) {
    if (storage.routed_sum) { return storage.routed_sum[std::int64_t(token) * kHidden + row]; }
    float value = 0;
    if (storage.contributions) {
#pragma unroll
        for (int j = 0; j < kTopK; ++j) {
            const int assignment = token * kTopK + j;
            const int column     = storage.groups.inverse[assignment];
            if (column >= 0) {
                value = fmaf(coefficients[assignment],
                             storage.contributions[std::int64_t(column) * kHidden + row], value);
            }
        }
    }
    return value;
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
