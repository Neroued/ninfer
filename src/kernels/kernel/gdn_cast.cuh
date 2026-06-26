#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__global__ void gdn_cast_bf16_to_f32_kernel(const __nv_bfloat16* in, float* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) { out[i] = __bfloat162float(in[i]); }
}

__global__ void gdn_cast_f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) { out[i] = __float2bfloat16(in[i]); }
}

} // namespace qus::kernels
