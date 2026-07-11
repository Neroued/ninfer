#pragma once

// qus::kernels - gdn_gating kernel: elementwise GDN gate prep over [48,T].
// Transcendentals use fp32 CUDA math functions, not polynomial approximations.

#include "kernels/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace qus::kernels {

__global__ void gdn_gating_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                  const float* A_log, const float* dt_bias, float* g,
                                  float* beta, std::int64_t n) {
    const std::int64_t start =
        blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const int h = static_cast<int>(i % 48);
        const float av = __bfloat162float(a[i]);
        const float bv = __bfloat162float(b[i]);
        const float sp = softplus(av + dt_bias[h]);
        g[i] = -expf(A_log[h]) * sp;
        beta[i] = sigmoid(bv);
    }
}

} // namespace qus::kernels
