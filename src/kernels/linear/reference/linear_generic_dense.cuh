#pragma once

// qus::kernels::detail - generic dense linear kernels. Dense q5090 payloads are raw row-major
// [N,K], so K is the fastest varying weight dimension. The dense GEMM path is intentionally simple
// and masked; dense control weights are correctness-only.

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {
namespace {

constexpr int kDenseGemvThreads   = 256;
constexpr int kDenseGemmBlockRows = 16;
constexpr int kDenseGemmBlockCols = 16;
constexpr int kDenseSmallGemvMaxN = 128;
constexpr int kDenseSmallGemvMaxT = 16;

template <int Width = 32>
__device__ __forceinline__ float dense_warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset, Width);
    }
    return v;
}

template <int BlockSize>
__device__ __forceinline__ float dense_block_reduce_sum(float v) {
    static_assert(BlockSize % 32 == 0, "BlockSize must be a whole number of warps");
    __shared__ float warp_sums[BlockSize / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = dense_warp_reduce_sum(v);
    if (lane == 0) { warp_sums[warp] = v; }
    __syncthreads();

    v = (threadIdx.x < (BlockSize / 32)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) { v = dense_warp_reduce_sum<BlockSize / 32>(v); }
    return v;
}

__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

} // namespace

__device__ __forceinline__ float load_dense_weight(const void* weight, std::int64_t index,
                                                   bool weight_fp32) {
    if (weight_fp32) { return static_cast<const float*>(weight)[index]; }
    return __bfloat162float(static_cast<const __nv_bfloat16*>(weight)[index]);
}

__global__ void linear_generic_dense_gemv_kernel(const __nv_bfloat16* x, const void* weight,
                                                 __nv_bfloat16* out, std::int32_t n,
                                                 std::int32_t k, bool weight_fp32) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
    if (row >= n) { return; }

    float acc                   = 0.0f;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * k;

    if ((k & 1) == 0) {
        const std::int32_t pairs = k >> 1;
        const auto* x2           = reinterpret_cast<const __nv_bfloat162*>(x);
        if (weight_fp32) {
            const auto* w2 =
                reinterpret_cast<const float2*>(static_cast<const float*>(weight) + row_base);
            for (std::int32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
                const float2 wf = w2[p];
                const float2 xf = bf162_to_float2(x2[p]);
                acc             = fmaf(wf.x, xf.x, acc);
                acc             = fmaf(wf.y, xf.y, acc);
            }
        } else {
            const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(
                static_cast<const __nv_bfloat16*>(weight) + row_base);
            for (std::int32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
                const float2 wf = bf162_to_float2(w2[p]);
                const float2 xf = bf162_to_float2(x2[p]);
                acc             = fmaf(wf.x, xf.x, acc);
                acc             = fmaf(wf.y, xf.y, acc);
            }
        }
    } else {
        for (std::int32_t kk = threadIdx.x; kk < k; kk += blockDim.x) {
            const float w  = load_dense_weight(weight, row_base + kk, weight_fp32);
            const float xv = __bfloat162float(x[kk]);
            acc            = fmaf(w, xv, acc);
        }
    }

    acc = dense_block_reduce_sum<kDenseGemvThreads>(acc);
    if (threadIdx.x == 0) { out[row] = __float2bfloat16(acc); }
}

__global__ void linear_generic_dense_small_gemv_kernel(const __nv_bfloat16* x, const void* weight,
                                                       __nv_bfloat16* out, std::int32_t n,
                                                       std::int32_t k, std::int32_t t,
                                                       bool weight_fp32) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t col = static_cast<std::int32_t>(blockIdx.y);
    if (row >= n || col >= t) { return; }

    float acc                    = 0.0f;
    const std::int64_t row_base  = static_cast<std::int64_t>(row) * k;
    const __nv_bfloat16* x_col   = x + static_cast<std::int64_t>(col) * k;
    const std::int64_t out_index = static_cast<std::int64_t>(col) * n + row;

    if ((k & 1) == 0) {
        const std::int32_t pairs = k >> 1;
        const auto* x2           = reinterpret_cast<const __nv_bfloat162*>(x_col);
        if (weight_fp32) {
            const auto* w2 =
                reinterpret_cast<const float2*>(static_cast<const float*>(weight) + row_base);
            for (std::int32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
                const float2 wf = w2[p];
                const float2 xf = bf162_to_float2(x2[p]);
                acc             = fmaf(wf.x, xf.x, acc);
                acc             = fmaf(wf.y, xf.y, acc);
            }
        } else {
            const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(
                static_cast<const __nv_bfloat16*>(weight) + row_base);
            for (std::int32_t p = threadIdx.x; p < pairs; p += blockDim.x) {
                const float2 wf = bf162_to_float2(w2[p]);
                const float2 xf = bf162_to_float2(x2[p]);
                acc             = fmaf(wf.x, xf.x, acc);
                acc             = fmaf(wf.y, xf.y, acc);
            }
        }
    } else {
        for (std::int32_t kk = threadIdx.x; kk < k; kk += blockDim.x) {
            const float w  = load_dense_weight(weight, row_base + kk, weight_fp32);
            const float xv = __bfloat162float(x_col[kk]);
            acc            = fmaf(w, xv, acc);
        }
    }

    acc = dense_block_reduce_sum<kDenseGemvThreads>(acc);
    if (threadIdx.x == 0) { out[out_index] = __float2bfloat16(acc); }
}

__global__ void linear_generic_dense_gemm_kernel(const __nv_bfloat16* x, const void* weight,
                                                 __nv_bfloat16* out, std::int32_t n,
                                                 std::int32_t k, std::int32_t t,
                                                 bool weight_fp32) {
    const std::int32_t row =
        static_cast<std::int32_t>(blockIdx.x) * blockDim.x + static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t col =
        static_cast<std::int32_t>(blockIdx.y) * blockDim.y + static_cast<std::int32_t>(threadIdx.y);
    if (row >= n || col >= t) { return; }

    float acc                   = 0.0f;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * k;
    const __nv_bfloat16* x_col  = x + static_cast<std::int64_t>(col) * k;

    if ((k & 1) == 0) {
        const std::int32_t pairs = k >> 1;
        const auto* x2           = reinterpret_cast<const __nv_bfloat162*>(x_col);
        if (weight_fp32) {
            const auto* w2 =
                reinterpret_cast<const float2*>(static_cast<const float*>(weight) + row_base);
            for (std::int32_t p = 0; p < pairs; ++p) {
                const float2 wf = w2[p];
                const float2 xf = bf162_to_float2(x2[p]);
                acc             = fmaf(wf.x, xf.x, acc);
                acc             = fmaf(wf.y, xf.y, acc);
            }
        } else {
            const auto* w2 = reinterpret_cast<const __nv_bfloat162*>(
                static_cast<const __nv_bfloat16*>(weight) + row_base);
            for (std::int32_t p = 0; p < pairs; ++p) {
                const float2 wf = bf162_to_float2(w2[p]);
                const float2 xf = bf162_to_float2(x2[p]);
                acc             = fmaf(wf.x, xf.x, acc);
                acc             = fmaf(wf.y, xf.y, acc);
            }
        }
    } else {
        for (std::int32_t kk = 0; kk < k; ++kk) {
            const float w  = load_dense_weight(weight, row_base + kk, weight_fp32);
            const float xv = __bfloat162float(x_col[kk]);
            acc            = fmaf(w, xv, acc);
        }
    }

    out[row + static_cast<std::int64_t>(n) * col] = __float2bfloat16(acc);
}

} // namespace qus::kernels::detail
