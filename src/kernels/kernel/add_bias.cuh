#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__global__ void add_bias_kernel(const __nv_bfloat16* bias, __nv_bfloat16* x, std::int32_t d,
                                std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float value = __bfloat162float(x[i]) + __bfloat162float(bias[i % d]);
        x[i]              = __float2bfloat16_rn(value);
    }
}

} // namespace qus::kernels
