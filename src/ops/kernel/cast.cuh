#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

__global__ void cast_fp32_to_bf16_kernel(const float* source, __nv_bfloat16* destination,
                                         std::int64_t count) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < count; i += stride) {
        destination[i] = __float2bfloat16_rn(source[i]);
    }
}

} // namespace ninfer::ops
