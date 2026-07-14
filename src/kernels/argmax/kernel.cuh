#pragma once

// ninfer::kernels - argmax kernel. One CUDA block handles one column and reduces
// over vocab; equal values keep the lowest vocab index.

#include <cuda_bf16.h>
#include <cstdint>
#include <climits>
#include <math_constants.h>

namespace ninfer::kernels {

inline constexpr int kArgmaxBlock = 256;
inline constexpr int kArgmaxItemsPerThread = 1;

__device__ __forceinline__ bool argmax_better(float value, std::int32_t index,
                                              float best_value, std::int32_t best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void argmax_warp_reduce(float& value, std::int32_t& index) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value = __shfl_down_sync(kMask, value, offset);
        const std::int32_t other_index = __shfl_down_sync(kMask, index, offset);
        if (argmax_better(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

__device__ __forceinline__ void argmax_block_reduce(float& value, std::int32_t& index) {
    __shared__ float warp_values[8];
    __shared__ std::int32_t warp_indices[8];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    argmax_warp_reduce(value, index);
    if (lane == 0) {
        warp_values[warp] = value;
        warp_indices[warp] = index;
    }
    __syncthreads();

    value = (lane < (blockDim.x >> 5)) ? warp_values[lane] : -CUDART_INF_F;
    index = (lane < (blockDim.x >> 5)) ? warp_indices[lane] : INT32_MAX;
    if (warp == 0) { argmax_warp_reduce(value, index); }
}

__launch_bounds__(kArgmaxBlock) __global__ void argmax_kernel(const __nv_bfloat16* logits,
                                                             std::int32_t* out,
                                                             std::int32_t valid_rows,
                                                             std::int32_t physical_rows) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;

    float best_value = __bfloat162float(logits[base]);
    std::int32_t best_index = 0;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows;
         v += blockDim.x) {
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

__launch_bounds__(kArgmaxBlock) __global__ void argmax_tiled_atomic_kernel(
    const __nv_bfloat16* logits, std::int32_t* out, std::int32_t valid_rows,
    std::int32_t physical_rows) {
    const std::int32_t t = static_cast<std::int32_t>(blockIdx.y);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const std::int32_t tile_start =
        static_cast<std::int32_t>(blockIdx.x) * blockDim.x * kArgmaxItemsPerThread;

    float best_value = -CUDART_INF_F;
    std::int32_t best_index = INT32_MAX;
#pragma unroll
    for (int item = 0; item < kArgmaxItemsPerThread; ++item) {
        const std::int32_t v = tile_start + threadIdx.x + item * blockDim.x;
        if (v < valid_rows) {
            const float value = __bfloat162float(logits[base + v]);
            if (argmax_better(value, v, best_value, best_index)) {
                best_value = value;
                best_index = v;
            }
        }
    }

    argmax_block_reduce(best_value, best_index);
    if (threadIdx.x != 0 || best_index == INT32_MAX) { return; }

    int current = out[t];
    while (true) {
        const float current_value = __bfloat162float(logits[base + current]);
        if (!argmax_better(best_value, best_index, current_value, current)) { break; }
        const int observed = atomicCAS(reinterpret_cast<int*>(out + t), current, best_index);
        if (observed == current) { break; }
        current = observed;
    }
}

} // namespace ninfer::kernels
