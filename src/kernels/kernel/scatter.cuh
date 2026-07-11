#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__global__ void scatter_kernel(const __nv_bfloat16* src, const std::int32_t* indices,
                               __nv_bfloat16* dst, std::int32_t d, std::int32_t columns,
                               std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t row     = static_cast<std::int32_t>(i % d);
        const std::int32_t col     = static_cast<std::int32_t>(i / d);
        const std::int32_t dst_col = indices[col];
        if (dst_col >= 0 && dst_col < columns) {
            dst[static_cast<std::int64_t>(dst_col) * d + row] = src[i];
        }
    }
}

} // namespace qus::kernels
