#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels {

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

} // namespace qus::kernels
