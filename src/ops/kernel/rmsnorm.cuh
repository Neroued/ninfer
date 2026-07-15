#pragma once

// ninfer::ops - RMSNorm kernels over contiguous BF16 rows.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

enum class RmsEpilogue {
    Offset,
    Plain,
    Gated,
};

template <RmsEpilogue Epilogue>
__device__ __forceinline__ float rmsnorm_epilogue(float x, float inv, float weight, float z) {
    if constexpr (Epilogue == RmsEpilogue::Offset) { weight += 1.0f; }
    float value = x * inv * weight;
    if constexpr (Epilogue == RmsEpilogue::Gated) { value *= silu(z); }
    return value;
}

// Fast geometry for D in {64, 128, 192, 256}. One warp owns one row, keeps the input in
// registers, and uses only warp shuffles for the reduction. Block is a scheduling choice rather
// than part of the row geometry.
template <RmsEpilogue Epilogue, int Block>
__launch_bounds__(Block) __global__
    void rmsnorm_warp_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                    const __nv_bfloat162* z, __nv_bfloat162* out, std::int32_t d,
                                    std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    constexpr int kWarpsPerBlock   = Block / kWarpSize;
    constexpr int kMaxPairsPerLane = 4;
    const int lane                 = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp                 = static_cast<int>(threadIdx.x) / kWarpSize;
    const std::int64_t row         = static_cast<std::int64_t>(blockIdx.x) * kWarpsPerBlock + warp;
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[kMaxPairsPerLane];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            values[k]       = x[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    sum       = warp_reduce_sum(sum);
    float inv = lane == 0 ? rsqrtf(sum / static_cast<float>(d) + eps) : 0.0f;
    inv       = __shfl_sync(kFullWarpMask, inv, 0);

#pragma unroll
    for (int k = 0; k < kMaxPairsPerLane; ++k) {
        const int pair = lane + k * kWarpSize;
        if (pair < pairs) {
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(weight[pair]);
            float2 zf{0.0f, 0.0f};
            if constexpr (Epilogue == RmsEpilogue::Gated) {
                zf = __bfloat1622float2(z[row_base + pair]);
            }
            out[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Fast geometry for wide rows. One CTA owns one row and keeps up to MaxPairsPerThread BF16x2
// values per lane. The launcher admits only widths evenly divisible by the CTA vector span.
template <RmsEpilogue Epilogue, int Block, int MaxPairsPerThread>
__launch_bounds__(Block) __global__
    void rmsnorm_cta_bf16x2_kernel(const __nv_bfloat162* x, const __nv_bfloat162* weight,
                                   const __nv_bfloat162* z, __nv_bfloat162* out, std::int32_t d,
                                   std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs             = d / 2;
    const int pairs_per_thread  = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            values[k]       = x[row_base + pair];
            const float2 xf = __bfloat1622float2(values[k]);
            sum += xf.x * xf.x + xf.y * xf.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) { inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps); }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair  = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xf = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(weight[pair]);
            float2 zf{0.0f, 0.0f};
            if constexpr (Epilogue == RmsEpilogue::Gated) {
                zf = __bfloat1622float2(z[row_base + pair]);
            }
            out[row_base + pair] =
                __floats2bfloat162_rn(rmsnorm_epilogue<Epilogue>(xf.x, inv, wf.x, zf.x),
                                      rmsnorm_epilogue<Epilogue>(xf.y, inv, wf.y, zf.y));
        }
    }
}

// Functional fallback outside the aligned fast domains. It intentionally favors a simple complete
// implementation over another family of shape-specific paths.
template <RmsEpilogue Epilogue>
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
        if constexpr (Epilogue == RmsEpilogue::Gated) { zv = __bfloat162float(z[index]); }
        out[index] = __float2bfloat16_rn(rmsnorm_epilogue<Epilogue>(xv, inv, wv, zv));
    }
}

} // namespace ninfer::ops
