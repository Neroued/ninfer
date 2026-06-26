#pragma once

// qus::kernels - argmax kernel. One CUDA block handles one column and reduces
// over vocab; equal values keep the lowest vocab index.

#include <cuda_bf16.h>
#include <cstdint>

namespace qus::kernels {

__device__ __forceinline__ bool argmax_better(float value, std::int32_t index,
                                              float best_value, std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__launch_bounds__(256) __global__ void argmax_kernel(const __nv_bfloat16* logits,
                                                     std::int32_t* out,
                                                     std::int32_t vocab) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * vocab;

    float best_value = __bfloat162float(logits[base]);
    std::int32_t best_index = 0;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < vocab; v += blockDim.x) {
        const float value = __bfloat162float(logits[base + v]);
        if (argmax_better(value, v, best_value, best_index)) {
            best_value = value;
            best_index = v;
        }
    }

    __shared__ float values[256];
    __shared__ std::int32_t indices[256];
    values[threadIdx.x] = best_value;
    indices[threadIdx.x] = best_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_value = values[threadIdx.x + stride];
            const std::int32_t other_index = indices[threadIdx.x + stride];
            if (argmax_better(other_value, other_index, values[threadIdx.x],
                              indices[threadIdx.x])) {
                values[threadIdx.x] = other_value;
                indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) { out[t] = indices[0]; }
}

} // namespace qus::kernels
