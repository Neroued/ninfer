#pragma once

// qus::kernels - partial NeoX RoPE kernel over q and k, in place.
// One thread handles one (tensor, head, token, rotary pair).

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace qus::kernels {

inline constexpr int kRopeHeadDim = 256;

__device__ __forceinline__ void rope_apply_one(__nv_bfloat16* data, std::int64_t base, int pair,
                                               int half, float c, float s) {
    const std::int64_t idx1 = base + static_cast<std::int64_t>(pair);
    const std::int64_t idx2 = base + static_cast<std::int64_t>(pair + half);
    const float x1 = __bfloat162float(data[idx1]);
    const float x2 = __bfloat162float(data[idx2]);
    data[idx1] = __float2bfloat16_rn(x1 * c - x2 * s);
    data[idx2] = __float2bfloat16_rn(x2 * c + x1 * s);
}

__global__ void rope_kernel(const std::int32_t* positions, __nv_bfloat16* q, __nv_bfloat16* k,
                            int rotary_dim, float theta, std::int32_t q_heads,
                            std::int32_t k_heads, std::int32_t T, std::int64_t total_pairs) {
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t half = static_cast<std::int64_t>(rotary_dim / 2);
    const std::int64_t q_pairs = static_cast<std::int64_t>(q_heads) *
                                 static_cast<std::int64_t>(T) * half;

    for (std::int64_t linear = start; linear < total_pairs; linear += stride) {
        const bool is_q = linear < q_pairs;
        const std::int64_t local = is_q ? linear : linear - q_pairs;
        const std::int32_t heads = is_q ? q_heads : k_heads;
        const int pair = static_cast<int>(local % half);
        const std::int64_t row = local / half;
        const std::int32_t h = static_cast<std::int32_t>(row % heads);
        const std::int32_t t = static_cast<std::int32_t>(row / heads);

        const float freq = powf(theta, -2.0f * static_cast<float>(pair) /
                                           static_cast<float>(rotary_dim));
        const float angle = static_cast<float>(positions[t]) * freq;
        const float c = cosf(angle);
        const float s = sinf(angle);
        const std::int64_t base =
            (static_cast<std::int64_t>(t) * heads + h) * static_cast<std::int64_t>(kRopeHeadDim);
        rope_apply_one(is_q ? q : k, base, pair, static_cast<int>(half), c, s);
    }
}

} // namespace qus::kernels
