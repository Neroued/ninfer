#pragma once

// qus::kernels - rmsnorm kernel. One CUDA block handles one row, reducing over ne[0].

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__device__ __forceinline__ float rmsnorm_silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__launch_bounds__(256) __global__
    void rmsnorm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                        const __nv_bfloat16* z, __nv_bfloat16* out, std::int32_t d,
                        std::int64_t rows, float eps, bool unit_offset) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    float sum = 0.0f;
    const std::int64_t d64 = static_cast<std::int64_t>(d);
    for (std::int64_t i = threadIdx.x; i < d64; i += blockDim.x) {
        const float xv = __bfloat162float(x[base + i]);
        sum += xv * xv;
    }

    __shared__ float scratch[256];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { scratch[threadIdx.x] += scratch[threadIdx.x + stride]; }
        __syncthreads();
    }

    const float inv = rsqrtf(scratch[0] / static_cast<float>(d) + eps);
    for (std::int64_t i = threadIdx.x; i < d64; i += blockDim.x) {
        const std::int64_t idx = base + i;
        float w = __bfloat162float(weight[i]);
        if (unit_offset) { w += 1.0f; }
        float v = __bfloat162float(x[idx]) * inv * w;
        if (z != nullptr) { v *= rmsnorm_silu_f32(__bfloat162float(z[idx])); }
        out[idx] = __float2bfloat16_rn(v);
    }
}

} // namespace qus::kernels
