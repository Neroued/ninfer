#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__global__ void gdn_cast_bf16_to_f32_kernel(const __nv_bfloat16* in, float* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) { out[i] = __bfloat162float(in[i]); }
}

__global__ void gdn_cast_qkv_bf16_to_f32_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                                const __nv_bfloat16* v, float* q_out, float* k_out,
                                                float* v_out, std::int64_t q_n, std::int64_t k_n,
                                                std::int64_t v_n) {
    const std::int64_t total  = q_n + k_n + v_n;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < total; i += stride) {
        if (i < q_n) {
            q_out[i] = __bfloat162float(q[i]);
        } else if (i < q_n + k_n) {
            const std::int64_t j = i - q_n;
            k_out[j]             = __bfloat162float(k[j]);
        } else {
            const std::int64_t j = i - q_n - k_n;
            v_out[j]             = __bfloat162float(v[j]);
        }
    }
}

__global__ void gdn_cast_f32_to_bf16_kernel(const float* in, __nv_bfloat16* out, std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) { out[i] = __float2bfloat16(in[i]); }
}

} // namespace qus::kernels
