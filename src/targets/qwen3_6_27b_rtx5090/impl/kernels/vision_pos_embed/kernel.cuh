#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::kernels {

template <int Block>
__launch_bounds__(Block) __global__
    void vision_pos_embed_add_d1152_kernel(const __nv_bfloat162* table, const std::int32_t* indices,
                                           const float* weights, __nv_bfloat162* x,
                                           std::int32_t patches, std::int32_t table_rows) {
    constexpr int d     = 1152;
    constexpr int pairs = d / 2;
    const int patch     = static_cast<int>(blockIdx.x);
    if (patch >= patches) { return; }
    __shared__ std::int32_t corner_indices[4];
    __shared__ float corner_weights[4];
    if (threadIdx.x < 4) {
        const std::int64_t control  = static_cast<std::int64_t>(patch) * 4 + threadIdx.x;
        corner_indices[threadIdx.x] = indices[control];
        corner_weights[threadIdx.x] = weights[control];
    }
    __syncthreads();

    const std::int64_t x_base = static_cast<std::int64_t>(patch) * pairs;
    for (int pair = static_cast<int>(threadIdx.x); pair < pairs; pair += Block) {
        float lo = 0.0f;
        float hi = 0.0f;
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            const int index = corner_indices[corner];
            if (index >= 0 && index < table_rows) {
                const float2 value =
                    __bfloat1622float2(table[static_cast<std::int64_t>(index) * pairs + pair]);
                lo += value.x * corner_weights[corner];
                hi += value.y * corner_weights[corner];
            }
        }
        const __nv_bfloat162 rounded  = __floats2bfloat162_rn(lo, hi);
        const __nv_bfloat162 residual = x[x_base + pair];
        x[x_base + pair] = __floats2bfloat162_rn(__low2float(residual) + __low2float(rounded),
                                                 __high2float(residual) + __high2float(rounded));
    }
}

__global__ void vision_pos_embed_add_kernel(const __nv_bfloat16* table, const std::int32_t* indices,
                                            const float* weights, __nv_bfloat16* x, std::int32_t d,
                                            std::int32_t patches, std::int32_t table_rows,
                                            std::int64_t n) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t linear = start; linear < n; linear += stride) {
        const std::int32_t row   = static_cast<std::int32_t>(linear % d);
        const std::int32_t patch = static_cast<std::int32_t>(linear / d);
        float position           = 0.0f;
#pragma unroll
        for (int corner = 0; corner < 4; ++corner) {
            const std::int64_t control = static_cast<std::int64_t>(patch) * 4 + corner;
            const std::int32_t index   = indices[control];
            if (index >= 0 && index < table_rows) {
                position += __bfloat162float(table[static_cast<std::int64_t>(index) * d + row]) *
                            weights[control];
            }
        }
        // HF rounds the interpolated position to BF16 before the residual add.
        const __nv_bfloat16 rounded = __float2bfloat16_rn(position);
        x[linear] = __float2bfloat16_rn(__bfloat162float(x[linear]) + __bfloat162float(rounded));
    }
}

} // namespace ninfer::kernels
