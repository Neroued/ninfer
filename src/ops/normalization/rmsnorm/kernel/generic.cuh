#pragma once
#include "ops/normalization/rmsnorm/kernel/epilogue.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail::rmsnorm {

// Complete scalar route for odd dimensions and backing without BF16x2 alignment.
template <Epilogue Kind>
__launch_bounds__(256) __global__
    void rmsnorm_generic_kernel(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                const __nv_bfloat16* z, __nv_bfloat16* out, std::int32_t d,
                                std::int64_t rows, float eps) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const std::int64_t base = row * static_cast<std::int64_t>(d);
    float sum               = 0.0f;
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
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
    for (std::int64_t i = threadIdx.x; i < static_cast<std::int64_t>(d); i += blockDim.x) {
        const std::int64_t index = base + i;
        const float xv           = __bfloat162float(x[index]);
        const float wv           = __bfloat162float(weight[i]);
        float zv                 = 0.0f;
        if constexpr (is_gated<Kind>) { zv = __bfloat162float(z[index]); }
        out[index] = __float2bfloat16_rn(apply_epilogue<Kind>(xv, inv, wv, zv));
    }
}

} // namespace ninfer::ops::detail::rmsnorm
