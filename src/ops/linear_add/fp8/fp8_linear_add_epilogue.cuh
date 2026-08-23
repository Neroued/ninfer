#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8AddResidualEpilogue {
    const __nv_bfloat16* residual;
    std::int32_t rows;

    __device__ __forceinline__ float apply(std::int32_t row, std::int32_t token,
                                           float value) const {
        return value + __bfloat162float(residual[static_cast<std::int64_t>(token) * rows + row]);
    }
};

} // namespace ninfer::ops::detail
