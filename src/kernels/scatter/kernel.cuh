#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::kernels {

__global__ void scatter_bf16x2_kernel(const __nv_bfloat162* src, const std::int32_t* indices,
                                      __nv_bfloat162* dst, std::int32_t pairs,
                                      std::int32_t columns) {
    const std::int32_t src_col = static_cast<std::int32_t>(blockIdx.x);
    __shared__ std::int32_t dst_col;
    if (threadIdx.x == 0) { dst_col = indices[src_col]; }
    __syncthreads();
    if (dst_col < 0 || dst_col >= columns) { return; }

    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * pairs;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * pairs;
    for (std::int32_t pair = static_cast<std::int32_t>(threadIdx.x); pair < pairs;
         pair += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + pair] = src[src_base + pair];
    }
}

__global__ void scatter_scalar_kernel(const __nv_bfloat16* src, const std::int32_t* indices,
                                      __nv_bfloat16* dst, std::int32_t d,
                                      std::int32_t columns) {
    const std::int32_t src_col = static_cast<std::int32_t>(blockIdx.x);
    __shared__ std::int32_t dst_col;
    if (threadIdx.x == 0) { dst_col = indices[src_col]; }
    __syncthreads();
    if (dst_col < 0 || dst_col >= columns) { return; }

    const std::int64_t src_base = static_cast<std::int64_t>(src_col) * d;
    const std::int64_t dst_base = static_cast<std::int64_t>(dst_col) * d;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < d;
         row += static_cast<std::int32_t>(blockDim.x)) {
        dst[dst_base + row] = src[src_base + row];
    }
}

} // namespace ninfer::kernels
