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
