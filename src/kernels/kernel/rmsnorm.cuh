#pragma once

// qus::kernels - rmsnorm kernel. One CUDA block handles one row, reducing over ne[0].

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__device__ __forceinline__ float rmsnorm_silu_f32(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ float rmsnorm_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
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

__launch_bounds__(512) __global__
    void rmsnorm_d5120_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                              __nv_bfloat16* out, std::int64_t rows, float eps,
                              bool unit_offset) {
    constexpr int kD = 5120;
    constexpr int kPairs = kD / 2;
    constexpr int kBlock = 512;
    constexpr int kPairsPerThread = kPairs / kBlock;

    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* weight2 = reinterpret_cast<const __nv_bfloat162*>(weight);
    auto* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    const std::int64_t base = row * static_cast<std::int64_t>(kPairs);

    __nv_bfloat162 xs[kPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const int j = threadIdx.x + k * kBlock;
        const __nv_bfloat162 xv = x2[base + j];
        const float2 xf = __bfloat1622float2(xv);
        xs[k] = xv;
        sum += xf.x * xf.x + xf.y * xf.y;
    }

    sum = rmsnorm_warp_sum(sum);
    __shared__ float warp_sums[16];
    if ((threadIdx.x & 31) == 0) { warp_sums[threadIdx.x >> 5] = sum; }
    __syncthreads();

    float block_sum = (threadIdx.x < 16) ? warp_sums[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) { block_sum = rmsnorm_warp_sum(block_sum); }

    __shared__ float inv_shared;
    if (threadIdx.x == 0) {
        inv_shared = rsqrtf(block_sum / static_cast<float>(kD) + eps);
    }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < kPairsPerThread; ++k) {
        const int j = threadIdx.x + k * kBlock;
        const float2 xf = __bfloat1622float2(xs[k]);
        float2 wf = __bfloat1622float2(weight2[j]);
        if (unit_offset) {
            wf.x += 1.0f;
            wf.y += 1.0f;
        }
        out2[base + j] = __floats2bfloat162_rn(xf.x * inv * wf.x, xf.y * inv * wf.y);
    }
}

} // namespace qus::kernels
