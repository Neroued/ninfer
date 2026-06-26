#pragma once

// qus::kernels - dense linear kernels. Correctness-baseline GEMV/GEMM with
// fp32 accumulation and BF16 output. Dense q5090 payloads are raw row-major
// [N,K], so K is the fastest varying weight dimension.

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

__device__ __forceinline__ float load_dense_weight(const void* weight, std::int64_t index,
                                                   bool weight_fp32) {
    if (weight_fp32) { return static_cast<const float*>(weight)[index]; }
    return __bfloat162float(static_cast<const __nv_bfloat16*>(weight)[index]);
}

__global__ void linear_dense_gemv_kernel(const __nv_bfloat16* x, const void* weight,
                                         __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                         bool weight_fp32) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t row = start; row < n; row += stride) {
        float acc = 0.0f;
        for (std::int32_t kk = 0; kk < k; ++kk) {
            const float w =
                load_dense_weight(weight, static_cast<std::int64_t>(row) * k + kk, weight_fp32);
            const float xv = __bfloat162float(x[kk]);
            acc            = fmaf(w, xv, acc);
        }
        out[row] = __float2bfloat16(acc);
    }
}

__global__ void linear_dense_gemm_kernel(const __nv_bfloat16* x, const void* weight,
                                         __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                         std::int32_t t, bool weight_fp32) {
    const std::int64_t elements = static_cast<std::int64_t>(n) * t;
    const std::int64_t start    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride   = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < elements; i += stride) {
        const std::int32_t col = static_cast<std::int32_t>(i / n);
        const std::int32_t row = static_cast<std::int32_t>(i - static_cast<std::int64_t>(col) * n);
        float acc              = 0.0f;
        for (std::int32_t kk = 0; kk < k; ++kk) {
            const float w =
                load_dense_weight(weight, static_cast<std::int64_t>(row) * k + kk, weight_fp32);
            const float xv = __bfloat162float(x[kk + static_cast<std::int64_t>(k) * col]);
            acc            = fmaf(w, xv, acc);
        }
        out[i] = __float2bfloat16(acc);
    }
}

} // namespace qus::kernels
