#pragma once

// qus::kernels - l2norm kernel. One warp handles one row, reducing over ne[0].

#include "kernels/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__launch_bounds__(512) __global__ void l2norm_kernel(const __nv_bfloat16* x, __nv_bfloat16* out,
                                                      std::int32_t d, std::int64_t rows,
                                                      float eps) {
    constexpr unsigned kFullWarpMask = 0xffffffffu;
    constexpr int kWarpSize = 32;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const std::int64_t row = (static_cast<std::int64_t>(blockIdx.x) * (blockDim.x / kWarpSize)) +
                             static_cast<std::int64_t>(warp);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    const std::int64_t d64 = static_cast<std::int64_t>(d);

    float sum = 0.0f;
    if (d <= 128) {
        const std::int64_t i0 = static_cast<std::int64_t>(lane);
        const std::int64_t i1 = i0 + kWarpSize;
        const std::int64_t i2 = i1 + kWarpSize;
        const std::int64_t i3 = i2 + kWarpSize;

        float v0 = 0.0f;
        float v1 = 0.0f;
        float v2 = 0.0f;
        float v3 = 0.0f;
        if (i0 < d64) { v0 = __bfloat162float(x[base + i0]); }
        if (i1 < d64) { v1 = __bfloat162float(x[base + i1]); }
        if (i2 < d64) { v2 = __bfloat162float(x[base + i2]); }
        if (i3 < d64) { v3 = __bfloat162float(x[base + i3]); }
        sum = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;

        sum = warp_reduce_sum(sum, kFullWarpMask);

        float inv = (lane == 0) ? rsqrtf(sum + eps) : 0.0f;
        inv = __shfl_sync(kFullWarpMask, inv, 0);
        if (i0 < d64) { out[base + i0] = __float2bfloat16_rn(v0 * inv); }
        if (i1 < d64) { out[base + i1] = __float2bfloat16_rn(v1 * inv); }
        if (i2 < d64) { out[base + i2] = __float2bfloat16_rn(v2 * inv); }
        if (i3 < d64) { out[base + i3] = __float2bfloat16_rn(v3 * inv); }
        return;
    }

    for (std::int64_t i = lane; i < d64; i += kWarpSize) {
        const float xv = __bfloat162float(x[base + i]);
        sum += xv * xv;
    }

    sum = warp_reduce_sum(sum, kFullWarpMask);

    float inv = (lane == 0) ? rsqrtf(sum + eps) : 0.0f;
    inv = __shfl_sync(kFullWarpMask, inv, 0);
    for (std::int64_t i = lane; i < d64; i += kWarpSize) {
        const std::int64_t idx = base + i;
        const float v = __bfloat162float(x[idx]) * inv;
        out[idx] = __float2bfloat16_rn(v);
    }
}

} // namespace qus::kernels
