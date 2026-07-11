#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

struct LayerNormMoments {
    float mean;
    float m2;
    int count;
};

__device__ __forceinline__ LayerNormMoments layer_norm_merge(LayerNormMoments a,
                                                             LayerNormMoments b) {
    if (b.count == 0) { return a; }
    if (a.count == 0) { return b; }
    const float delta = b.mean - a.mean;
    const int count   = a.count + b.count;
    const float ratio = static_cast<float>(b.count) / static_cast<float>(count);
    return LayerNormMoments{
        a.mean + delta * ratio,
        a.m2 + b.m2 +
            delta * delta *
                (static_cast<float>(a.count) * static_cast<float>(b.count) /
                 static_cast<float>(count)),
        count,
    };
}

__device__ __forceinline__ LayerNormMoments layer_norm_warp_reduce(LayerNormMoments value) {
    constexpr unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const LayerNormMoments other{
            __shfl_down_sync(mask, value.mean, offset),
            __shfl_down_sync(mask, value.m2, offset),
            __shfl_down_sync(mask, value.count, offset),
        };
        value = layer_norm_merge(value, other);
    }
    return value;
}

__device__ __forceinline__ void layer_norm_update(LayerNormMoments& moments, float value) {
    ++moments.count;
    const float delta = value - moments.mean;
    moments.mean += delta / static_cast<float>(moments.count);
    moments.m2 += delta * (value - moments.mean);
}

template <int Block>
__launch_bounds__(Block) __global__
    void layer_norm_d1152_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                        const __nv_bfloat162* bias, __nv_bfloat162* out,
                                        std::int64_t rows, float eps) {
    constexpr int d        = 1152;
    constexpr int pairs    = d / 2;
    constexpr int warps    = Block / 32;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const std::int64_t base = row * pairs;

    LayerNormMoments local{0.0f, 0.0f, 0};
    for (int pair = static_cast<int>(threadIdx.x); pair < pairs; pair += Block) {
        const float2 values = __bfloat1622float2(x[base + pair]);
        layer_norm_update(local, values.x);
        layer_norm_update(local, values.y);
    }
    local = layer_norm_warp_reduce(local);

    __shared__ LayerNormMoments warp_moments[warps];
    __shared__ float final_mean;
    __shared__ float final_inv;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { warp_moments[warp] = local; }
    __syncthreads();

    if (warp == 0) {
        LayerNormMoments block_value =
            lane < warps ? warp_moments[lane] : LayerNormMoments{0.0f, 0.0f, 0};
        block_value = layer_norm_warp_reduce(block_value);
        if (lane == 0) {
            final_mean = block_value.mean;
            final_inv  = rsqrtf(block_value.m2 / static_cast<float>(d) + eps);
        }
    }
    __syncthreads();

    for (int pair = static_cast<int>(threadIdx.x); pair < pairs; pair += Block) {
        const float2 values = __bfloat1622float2(x[base + pair]);
        const float2 scales = __bfloat1622float2(weight[pair]);
        const float2 shifts = __bfloat1622float2(bias[pair]);
        out[base + pair] =
            __floats2bfloat162_rn((values.x - final_mean) * final_inv * scales.x + shifts.x,
                                  (values.y - final_mean) * final_inv * scales.y + shifts.y);
    }
}

template <int Block>
__launch_bounds__(Block) __global__
    void layer_norm_d1152_warp_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                      const __nv_bfloat162* bias, __nv_bfloat162* out,
                                      std::int64_t rows, float eps) {
    constexpr int d        = 1152;
    constexpr int pairs    = d / 2;
    constexpr int warps    = Block / 32;
    const int lane         = static_cast<int>(threadIdx.x) & 31;
    const int warp         = static_cast<int>(threadIdx.x) >> 5;
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * warps + warp;
    if (row >= rows) { return; }
    const std::int64_t base = row * pairs;

    LayerNormMoments local{0.0f, 0.0f, 0};
    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 values = __bfloat1622float2(x[base + pair]);
        layer_norm_update(local, values.x);
        layer_norm_update(local, values.y);
    }
    local                   = layer_norm_warp_reduce(local);
    constexpr unsigned mask = 0xffffffffu;
    const float mean        = __shfl_sync(mask, local.mean, 0);
    const float m2          = __shfl_sync(mask, local.m2, 0);
    const float inv         = rsqrtf(m2 / static_cast<float>(d) + eps);

    for (int pair = lane; pair < pairs; pair += 32) {
        const float2 values = __bfloat1622float2(x[base + pair]);
        const float2 scales = __bfloat1622float2(weight[pair]);
        const float2 shifts = __bfloat1622float2(bias[pair]);
        out[base + pair]    = __floats2bfloat162_rn((values.x - mean) * inv * scales.x + shifts.x,
                                                    (values.y - mean) * inv * scales.y + shifts.y);
    }
}

template <int Block>
__global__ void layer_norm_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                  const __nv_bfloat16* bias, __nv_bfloat16* out, std::int32_t d,
                                  std::int64_t rows, float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }
    const std::int64_t base = row * d;

    LayerNormMoments local{0.0f, 0.0f, 0};
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < d; i += Block) {
        const float value = __bfloat162float(x[base + i]);
        ++local.count;
        const float delta = value - local.mean;
        local.mean += delta / static_cast<float>(local.count);
        local.m2 += delta * (value - local.mean);
    }

    __shared__ float means[Block];
    __shared__ float m2s[Block];
    __shared__ int counts[Block];
    means[threadIdx.x]  = local.mean;
    m2s[threadIdx.x]    = local.m2;
    counts[threadIdx.x] = local.count;
    __syncthreads();

    for (int offset = Block / 2; offset > 0; offset >>= 1) {
        if (static_cast<int>(threadIdx.x) < offset) {
            const LayerNormMoments a{means[threadIdx.x], m2s[threadIdx.x], counts[threadIdx.x]};
            const LayerNormMoments b{means[threadIdx.x + offset], m2s[threadIdx.x + offset],
                                     counts[threadIdx.x + offset]};
            const LayerNormMoments merged = layer_norm_merge(a, b);
            means[threadIdx.x]            = merged.mean;
            m2s[threadIdx.x]              = merged.m2;
            counts[threadIdx.x]           = merged.count;
        }
        __syncthreads();
    }

    const float mean = means[0];
    const float inv  = rsqrtf(m2s[0] / static_cast<float>(d) + eps);
    for (std::int32_t i = static_cast<std::int32_t>(threadIdx.x); i < d; i += Block) {
        const float value = (__bfloat162float(x[base + i]) - mean) * inv;
        out[base + i] =
            __float2bfloat16_rn(value * __bfloat162float(weight[i]) + __bfloat162float(bias[i]));
    }
}

} // namespace qus::kernels
