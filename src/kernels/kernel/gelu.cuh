#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

template <bool TanhApprox>
__global__ void gelu_kernel(__nv_bfloat16* x, std::int64_t n) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    constexpr float kSqrt2Pi  = 0.79788456080286535588f;
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float value = __bfloat162float(x[i]);
        float out;
        if constexpr (TanhApprox) {
            out = 0.5f * value *
                  (1.0f + tanhf(kSqrt2Pi * (value + 0.044715f * value * value * value)));
        } else {
            out = 0.5f * value * (1.0f + erff(value * kInvSqrt2));
        }
        x[i] = __float2bfloat16_rn(out);
    }
}

} // namespace qus::kernels
