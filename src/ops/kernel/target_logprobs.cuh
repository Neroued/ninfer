#pragma once

// Implements: include/ninfer/ops/target_logprobs.h
// Match: contiguous BF16 [physical_rows,C], I32 [C], and FP32 [C].
// Algorithm assumptions: one 256-thread CTA performs a stable two-pass logsumexp per column.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kTargetLogprobsBlock = 256;

template <int BlockSize>
__device__ __forceinline__ float target_logprobs_block_max(float value) {
    static_assert(BlockSize >= kWarpSize && BlockSize <= 1024);
    static_assert((BlockSize & (BlockSize - 1)) == 0);
    constexpr int kWarps = BlockSize / kWarpSize;
    __shared__ float warp_maxima[kWarps];
    __shared__ float result;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    value          = warp_max(value);
    if (lane == 0) { warp_maxima[warp] = value; }
    __syncthreads();

    if (warp == 0) {
        value = lane < kWarps ? warp_maxima[lane] : -CUDART_INF_F;
        value = warp_max(value);
        if (lane == 0) { result = value; }
    }
    __syncthreads();
    return result;
}

template <int BlockSize>
__launch_bounds__(BlockSize) __global__
    void target_logprobs_kernel(const __nv_bfloat16* logits, const std::int32_t* target_ids,
                                float* output, std::int32_t valid_rows,
                                std::int32_t physical_rows) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base   = static_cast<std::int64_t>(column) * physical_rows;

    float local_max = -CUDART_INF_F;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        local_max = fmaxf(local_max, __bfloat162float(logits[base + row]));
    }
    const float maximum = target_logprobs_block_max<BlockSize>(local_max);

    float local_sum = 0.0f;
    for (std::int32_t row = static_cast<std::int32_t>(threadIdx.x); row < valid_rows;
         row += BlockSize) {
        local_sum += expf(__bfloat162float(logits[base + row]) - maximum);
    }
    __shared__ float warp_sums[BlockSize / kWarpSize];
    const float sum = block_reduce_sum<BlockSize>(local_sum, warp_sums);
    if (threadIdx.x == 0) {
        const float target = __bfloat162float(logits[base + target_ids[column]]);
        output[column]     = target - maximum - logf(sum);
    }
}

} // namespace ninfer::ops
