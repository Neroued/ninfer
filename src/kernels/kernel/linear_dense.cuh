#pragma once

// qus::kernels - dense linear kernels. Dense q5090 payloads are raw row-major
// [N,K], so K is the fastest varying weight dimension. BF16 GEMM uses BF16
// tensor cores with fp32 accumulation. FP32_CTRL GEMM uses TF32 tensor cores so
// T>1 keeps more control-weight precision than the BF16_CTRL path.

#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>

namespace qus::kernels {
namespace {

constexpr int kDenseMmaM            = 16;
constexpr int kDenseMmaN            = 16;
constexpr int kDenseMmaK            = 16;
constexpr int kDenseTf32MmaK        = 8;
constexpr int kDenseGemmWarps       = 8;
constexpr int kDenseGemmThreads     = kDenseGemmWarps * 32;
constexpr int kDenseGemvThreads     = 256;
constexpr int kDenseGemvWarps       = kDenseGemvThreads / 32;
constexpr int kDenseMmaTileElements = kDenseMmaM * kDenseMmaK;
constexpr int kDenseGemmBlockM      = 64;
constexpr int kDenseGemmBlockN      = 32;
constexpr int kDenseGemmBlockK      = 64;
constexpr int kDenseGemmWarpRows    = kDenseGemmBlockM / kDenseMmaM;
constexpr int kDenseGemmWarpCols    = kDenseGemmBlockN / kDenseMmaN;
constexpr int kDenseGemmBlockA      = kDenseGemmBlockM * kDenseGemmBlockK;
constexpr int kDenseGemmBlockB      = kDenseGemmBlockK * kDenseGemmBlockN;
constexpr int kDenseGemmWideBlockN  = 64;
constexpr int kDenseGemmWideAccCols = 2;
constexpr int kDenseGemmWideWarpColGroups =
    kDenseGemmWideBlockN / (kDenseMmaN * kDenseGemmWideAccCols);
constexpr int kDenseGemmHalfBlockM     = 32;
constexpr int kDenseGemmHalfBlockK     = 96;
constexpr int kDenseGemmHalfWarps      = 4;
constexpr int kDenseGemmHalfThreads    = kDenseGemmHalfWarps * 32;
constexpr int kDenseGemmHalfBlockA     = kDenseGemmHalfBlockM * kDenseGemmHalfBlockK;
constexpr int kDenseGemmHalfWideBlockB = kDenseGemmHalfBlockK * kDenseGemmWideBlockN;
constexpr int kDenseSmallGemvMaxN      = 128;
constexpr int kDenseSmallGemvMaxT      = 16;

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

__device__ __forceinline__ __nv_bfloat16 dense_zero_bf16() { return __float2bfloat16(0.0f); }

__device__ __forceinline__ void dense_cp_async_16(void* smem_dst, const void* gmem_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                     static_cast<unsigned>(__cvta_generic_to_shared(smem_dst))),
                 "l"(gmem_src));
#else
    *reinterpret_cast<int4*>(smem_dst) = *reinterpret_cast<const int4*>(gmem_src);
#endif
}

__device__ __forceinline__ void dense_cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
#endif
}

__device__ __forceinline__ void dense_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;\n");
#endif
}

template <int BlockM, int BlockN, int BlockK>
__device__ __forceinline__ void
dense_gemm_issue_bf16_full_tile(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                                __nv_bfloat16* smem_a, __nv_bfloat16* smem_b, std::int32_t k,
                                std::int32_t row0, std::int32_t col0, std::int32_t k0) {
    constexpr int kVecElems = 8; // int4 holds eight bf16 values.
    constexpr int kBlockA   = BlockM * BlockK;
    constexpr int kBlockB   = BlockK * BlockN;

    auto* smem_a_vec = reinterpret_cast<int4*>(smem_a);
    for (int v = threadIdx.x; v < kBlockA / kVecElems; v += blockDim.x) {
        const int i              = v * kVecElems;
        const std::int32_t r     = i / BlockK;
        const std::int32_t kk    = i - r * BlockK;
        const std::int64_t g_idx = static_cast<std::int64_t>(row0 + r) * k + k0 + kk;
        dense_cp_async_16(smem_a_vec + v, weight + g_idx);
    }

    auto* smem_b_vec = reinterpret_cast<int4*>(smem_b);
    for (int v = threadIdx.x; v < kBlockB / kVecElems; v += blockDim.x) {
        const int i              = v * kVecElems;
        const std::int32_t c     = i / BlockK;
        const std::int32_t kk    = i - c * BlockK;
        const std::int64_t g_idx = k0 + kk + static_cast<std::int64_t>(k) * (col0 + c);
        dense_cp_async_16(smem_b_vec + v, x + g_idx);
    }
}

} // namespace

__device__ __forceinline__ float load_dense_weight(const void* weight, std::int64_t index,
                                                   bool weight_fp32) {
    if (weight_fp32) { return static_cast<const float*>(weight)[index]; }
    return __bfloat162float(static_cast<const __nv_bfloat16*>(weight)[index]);
}

__global__ void linear_dense_gemv_kernel(const __nv_bfloat16* x, const void* weight,
                                         __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                         bool weight_fp32) {
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

__global__ void linear_dense_small_gemv_kernel(const __nv_bfloat16* x, const void* weight,
                                               __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                               std::int32_t t, bool weight_fp32) {
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

__global__ void linear_dense_gemm_kernel(const __nv_bfloat16* x, const void* weight,
                                         __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                         std::int32_t t, bool weight_fp32) {
    using namespace nvcuda;

    __shared__ __align__(16) __nv_bfloat16 smem_a[kDenseGemmBlockA];
    __shared__ __align__(16) __nv_bfloat16 smem_b[kDenseGemmBlockB];
    __shared__ __align__(16) float smem_c[kDenseGemmWarps][kDenseMmaTileElements];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int warp_m             = warp / kDenseGemmWarpCols;
    const int warp_n             = warp - warp_m * kDenseGemmWarpCols;
    const std::int32_t row0      = static_cast<std::int32_t>(blockIdx.x) * kDenseGemmBlockM;
    const std::int32_t col0      = static_cast<std::int32_t>(blockIdx.y) * kDenseGemmBlockN;
    const std::int32_t warp_row0 = row0 + warp_m * kDenseMmaM;
    const std::int32_t warp_col0 = col0 + warp_n * kDenseMmaN;

    wmma::fragment<wmma::matrix_a, kDenseMmaM, kDenseMmaN, kDenseMmaK, __nv_bfloat16,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kDenseMmaM, kDenseMmaN, kDenseMmaK, __nv_bfloat16,
                   wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, kDenseMmaM, kDenseMmaN, kDenseMmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (std::int32_t k0 = 0; k0 < k; k0 += kDenseGemmBlockK) {
        const bool full_a           = row0 + kDenseGemmBlockM <= n && k0 + kDenseGemmBlockK <= k;
        const bool full_b           = col0 + kDenseGemmBlockN <= t && k0 + kDenseGemmBlockK <= k;
        const bool bf16_vec_aligned = (k & 7) == 0;
        const bool fp32_vec_aligned = (k & 3) == 0;

        if (!weight_fp32 && full_a && bf16_vec_aligned) {
            constexpr int kVecElems = 8; // int4 holds eight bf16 values.
            const auto* w_bf16      = static_cast<const __nv_bfloat16*>(weight);
            auto* smem_a_vec        = reinterpret_cast<int4*>(smem_a);
            for (int v = threadIdx.x; v < kDenseGemmBlockA / kVecElems; v += blockDim.x) {
                const int i              = v * kVecElems;
                const std::int32_t r     = i / kDenseGemmBlockK;
                const std::int32_t kk    = i - r * kDenseGemmBlockK;
                const std::int64_t g_idx = static_cast<std::int64_t>(row0 + r) * k + k0 + kk;
                smem_a_vec[v]            = *reinterpret_cast<const int4*>(w_bf16 + g_idx);
            }
        } else if (weight_fp32 && full_a && fp32_vec_aligned) {
            constexpr int kVecElems = 4; // float4 source converted to four bf16 values.
            const auto* w_f32       = static_cast<const float*>(weight);
            for (int v = threadIdx.x; v < kDenseGemmBlockA / kVecElems; v += blockDim.x) {
                const int i              = v * kVecElems;
                const std::int32_t r     = i / kDenseGemmBlockK;
                const std::int32_t kk    = i - r * kDenseGemmBlockK;
                const std::int64_t g_idx = static_cast<std::int64_t>(row0 + r) * k + k0 + kk;
                const float4 wv          = *reinterpret_cast<const float4*>(w_f32 + g_idx);
                smem_a[i]                = __float2bfloat16(wv.x);
                smem_a[i + 1]            = __float2bfloat16(wv.y);
                smem_a[i + 2]            = __float2bfloat16(wv.z);
                smem_a[i + 3]            = __float2bfloat16(wv.w);
            }
        } else {
            for (int i = threadIdx.x; i < kDenseGemmBlockA; i += blockDim.x) {
                const std::int32_t r   = i / kDenseGemmBlockK;
                const std::int32_t kk  = i - r * kDenseGemmBlockK;
                const std::int32_t row = row0 + r;
                const std::int32_t col = k0 + kk;
                __nv_bfloat16 v        = dense_zero_bf16();
                if (row < n && col < k) {
                    const std::int64_t idx = static_cast<std::int64_t>(row) * k + col;
                    v = weight_fp32 ? __float2bfloat16(static_cast<const float*>(weight)[idx])
                                    : static_cast<const __nv_bfloat16*>(weight)[idx];
                }
                smem_a[i] = v;
            }
        }

        if (full_b && bf16_vec_aligned) {
            constexpr int kVecElems = 8; // int4 holds eight bf16 values.
            auto* smem_b_vec        = reinterpret_cast<int4*>(smem_b);
            for (int v = threadIdx.x; v < kDenseGemmBlockB / kVecElems; v += blockDim.x) {
                const int i              = v * kVecElems;
                const std::int32_t c     = i / kDenseGemmBlockK;
                const std::int32_t kk    = i - c * kDenseGemmBlockK;
                const std::int64_t g_idx = k0 + kk + static_cast<std::int64_t>(k) * (col0 + c);
                smem_b_vec[v]            = *reinterpret_cast<const int4*>(x + g_idx);
            }
        } else {
            for (int i = threadIdx.x; i < kDenseGemmBlockB; i += blockDim.x) {
                const std::int32_t c   = i / kDenseGemmBlockK;
                const std::int32_t kk  = i - c * kDenseGemmBlockK;
                const std::int32_t row = k0 + kk;
                const std::int32_t col = col0 + c;
                __nv_bfloat16 v        = dense_zero_bf16();
                if (row < k && col < t) { v = x[row + static_cast<std::int64_t>(k) * col]; }
                smem_b[i] = v;
            }
        }
        __syncthreads();

        for (int kk = 0; kk < kDenseGemmBlockK; kk += kDenseMmaK) {
            wmma::load_matrix_sync(a_frag, smem_a + warp_m * kDenseMmaM * kDenseGemmBlockK + kk,
                                   kDenseGemmBlockK);
            wmma::load_matrix_sync(b_frag, smem_b + warp_n * kDenseMmaN * kDenseGemmBlockK + kk,
                                   kDenseGemmBlockK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    float* tile_c = smem_c[warp];
    wmma::store_matrix_sync(tile_c, c_frag, kDenseMmaM, wmma::mem_col_major);
    __syncwarp();

    for (int i = lane; i < kDenseMmaTileElements; i += 32) {
        const std::int32_t r   = i & (kDenseMmaM - 1);
        const std::int32_t c   = i >> 4;
        const std::int32_t row = warp_row0 + r;
        const std::int32_t col = warp_col0 + c;
        if (row < n && col < t) {
            out[row + static_cast<std::int64_t>(n) * col] = __float2bfloat16(tile_c[i]);
        }
    }
}

__global__ void linear_dense_gemm_tf32_kernel(const __nv_bfloat16* x, const float* weight,
                                              __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                              std::int32_t t) {
    using namespace nvcuda;

    __shared__ __align__(16) float smem_a[kDenseGemmBlockA];
    __shared__ __align__(16) float smem_b[kDenseGemmBlockB];
    __shared__ __align__(16) float smem_c[kDenseGemmWarps][kDenseMmaTileElements];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int warp_m             = warp / kDenseGemmWarpCols;
    const int warp_n             = warp - warp_m * kDenseGemmWarpCols;
    const std::int32_t row0      = static_cast<std::int32_t>(blockIdx.x) * kDenseGemmBlockM;
    const std::int32_t col0      = static_cast<std::int32_t>(blockIdx.y) * kDenseGemmBlockN;
    const std::int32_t warp_row0 = row0 + warp_m * kDenseMmaM;
    const std::int32_t warp_col0 = col0 + warp_n * kDenseMmaN;

    wmma::fragment<wmma::matrix_a, kDenseMmaM, kDenseMmaN, kDenseTf32MmaK, wmma::precision::tf32,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kDenseMmaM, kDenseMmaN, kDenseTf32MmaK, wmma::precision::tf32,
                   wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, kDenseMmaM, kDenseMmaN, kDenseTf32MmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (std::int32_t k0 = 0; k0 < k; k0 += kDenseGemmBlockK) {
        for (int i = threadIdx.x; i < kDenseGemmBlockA; i += blockDim.x) {
            const std::int32_t r   = i / kDenseGemmBlockK;
            const std::int32_t kk  = i - r * kDenseGemmBlockK;
            const std::int32_t row = row0 + r;
            const std::int32_t col = k0 + kk;
            float v                = 0.0f;
            if (row < n && col < k) { v = weight[static_cast<std::int64_t>(row) * k + col]; }
            smem_a[i] = v;
        }

        for (int i = threadIdx.x; i < kDenseGemmBlockB; i += blockDim.x) {
            const std::int32_t c   = i / kDenseGemmBlockK;
            const std::int32_t kk  = i - c * kDenseGemmBlockK;
            const std::int32_t row = k0 + kk;
            const std::int32_t col = col0 + c;
            float v                = 0.0f;
            if (row < k && col < t) {
                v = __bfloat162float(x[row + static_cast<std::int64_t>(k) * col]);
            }
            smem_b[i] = v;
        }
        __syncthreads();

        for (int kk = 0; kk < kDenseGemmBlockK; kk += kDenseTf32MmaK) {
            wmma::load_matrix_sync(a_frag, smem_a + warp_m * kDenseMmaM * kDenseGemmBlockK + kk,
                                   kDenseGemmBlockK);
            wmma::load_matrix_sync(b_frag, smem_b + warp_n * kDenseMmaN * kDenseGemmBlockK + kk,
                                   kDenseGemmBlockK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }

    float* tile_c = smem_c[warp];
    wmma::store_matrix_sync(tile_c, c_frag, kDenseMmaM, wmma::mem_col_major);
    __syncwarp();

    for (int i = lane; i < kDenseMmaTileElements; i += 32) {
        const std::int32_t r   = i & (kDenseMmaM - 1);
        const std::int32_t c   = i >> 4;
        const std::int32_t row = warp_row0 + r;
        const std::int32_t col = warp_col0 + c;
        if (row < n && col < t) {
            out[row + static_cast<std::int64_t>(n) * col] = __float2bfloat16(tile_c[i]);
        }
    }
}

__global__ void linear_dense_gemm_bf16_full_kernel(const __nv_bfloat16* x,
                                                   const __nv_bfloat16* weight, __nv_bfloat16* out,
                                                   std::int32_t n, std::int32_t k) {
    using namespace nvcuda;

    __shared__ __align__(16) __nv_bfloat16 smem_a[2][kDenseGemmBlockA];
    __shared__ __align__(16) __nv_bfloat16 smem_b[2][kDenseGemmBlockB];
    __shared__ __align__(16) float smem_c[kDenseGemmWarps][kDenseMmaTileElements];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int warp_m             = warp / kDenseGemmWarpCols;
    const int warp_n             = warp - warp_m * kDenseGemmWarpCols;
    const std::int32_t row0      = static_cast<std::int32_t>(blockIdx.x) * kDenseGemmBlockM;
    const std::int32_t col0      = static_cast<std::int32_t>(blockIdx.y) * kDenseGemmBlockN;
    const std::int32_t warp_row0 = row0 + warp_m * kDenseMmaM;
    const std::int32_t warp_col0 = col0 + warp_n * kDenseMmaN;

    wmma::fragment<wmma::matrix_a, kDenseMmaM, kDenseMmaN, kDenseMmaK, __nv_bfloat16,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kDenseMmaM, kDenseMmaN, kDenseMmaK, __nv_bfloat16,
                   wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, kDenseMmaM, kDenseMmaN, kDenseMmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    dense_gemm_issue_bf16_full_tile<kDenseGemmBlockM, kDenseGemmBlockN, kDenseGemmBlockK>(
        x, weight, smem_a[0], smem_b[0], k, row0, col0, 0);
    dense_cp_async_commit();
    dense_cp_async_wait_all();
    __syncthreads();

    int stage = 0;
    for (std::int32_t k0 = 0; k0 < k; k0 += kDenseGemmBlockK) {
        const std::int32_t next_k0 = k0 + kDenseGemmBlockK;
        const int next_stage       = stage ^ 1;
        if (next_k0 < k) {
            dense_gemm_issue_bf16_full_tile<kDenseGemmBlockM, kDenseGemmBlockN, kDenseGemmBlockK>(
                x, weight, smem_a[next_stage], smem_b[next_stage], k, row0, col0, next_k0);
            dense_cp_async_commit();
        }

        for (int kk = 0; kk < kDenseGemmBlockK; kk += kDenseMmaK) {
            wmma::load_matrix_sync(a_frag,
                                   smem_a[stage] + warp_m * kDenseMmaM * kDenseGemmBlockK + kk,
                                   kDenseGemmBlockK);
            wmma::load_matrix_sync(b_frag,
                                   smem_b[stage] + warp_n * kDenseMmaN * kDenseGemmBlockK + kk,
                                   kDenseGemmBlockK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }

        if (next_k0 < k) { dense_cp_async_wait_all(); }
        __syncthreads();
        stage = next_stage;
    }

    float* tile_c = smem_c[warp];
    wmma::store_matrix_sync(tile_c, c_frag, kDenseMmaM, wmma::mem_col_major);
    __syncwarp();

    for (int i = lane; i < kDenseMmaTileElements; i += 32) {
        const std::int32_t r                          = i & (kDenseMmaM - 1);
        const std::int32_t c                          = i >> 4;
        const std::int32_t row                        = warp_row0 + r;
        const std::int32_t col                        = warp_col0 + c;
        out[row + static_cast<std::int64_t>(n) * col] = __float2bfloat16(tile_c[i]);
    }
}

__global__ void linear_dense_gemm_bf16_half_wide_full_kernel(const __nv_bfloat16* x,
                                                             const __nv_bfloat16* weight,
                                                             __nv_bfloat16* out, std::int32_t n,
                                                             std::int32_t k) {
    using namespace nvcuda;

    __shared__ __align__(16) __nv_bfloat16 smem_a[2][kDenseGemmHalfBlockA];
    __shared__ __align__(16) __nv_bfloat16 smem_b[2][kDenseGemmHalfWideBlockB];
    __shared__ __align__(
        16) float smem_c[kDenseGemmHalfWarps][kDenseGemmWideAccCols][kDenseMmaTileElements];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const int warp_m             = warp / kDenseGemmWideWarpColGroups;
    const int warp_n_group       = warp - warp_m * kDenseGemmWideWarpColGroups;
    const std::int32_t row0      = static_cast<std::int32_t>(blockIdx.x) * kDenseGemmHalfBlockM;
    const std::int32_t col0      = static_cast<std::int32_t>(blockIdx.y) * kDenseGemmWideBlockN;
    const std::int32_t warp_row0 = row0 + warp_m * kDenseMmaM;
    const std::int32_t warp_col0 = col0 + warp_n_group * kDenseGemmWideAccCols * kDenseMmaN;

    wmma::fragment<wmma::matrix_a, kDenseMmaM, kDenseMmaN, kDenseMmaK, __nv_bfloat16,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kDenseMmaM, kDenseMmaN, kDenseMmaK, __nv_bfloat16,
                   wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, kDenseMmaM, kDenseMmaN, kDenseMmaK, float>
        c_frag[kDenseGemmWideAccCols];
#pragma unroll
    for (int c = 0; c < kDenseGemmWideAccCols; ++c) { wmma::fill_fragment(c_frag[c], 0.0f); }

    dense_gemm_issue_bf16_full_tile<kDenseGemmHalfBlockM, kDenseGemmWideBlockN,
                                    kDenseGemmHalfBlockK>(x, weight, smem_a[0], smem_b[0], k, row0,
                                                          col0, 0);
    dense_cp_async_commit();
    dense_cp_async_wait_all();
    __syncthreads();

    int stage = 0;
    for (std::int32_t k0 = 0; k0 < k; k0 += kDenseGemmHalfBlockK) {
        const std::int32_t next_k0 = k0 + kDenseGemmHalfBlockK;
        const int next_stage       = stage ^ 1;
        if (next_k0 < k) {
            dense_gemm_issue_bf16_full_tile<kDenseGemmHalfBlockM, kDenseGemmWideBlockN,
                                            kDenseGemmHalfBlockK>(
                x, weight, smem_a[next_stage], smem_b[next_stage], k, row0, col0, next_k0);
            dense_cp_async_commit();
        }

        for (int kk = 0; kk < kDenseGemmHalfBlockK; kk += kDenseMmaK) {
            wmma::load_matrix_sync(a_frag,
                                   smem_a[stage] + warp_m * kDenseMmaM * kDenseGemmHalfBlockK + kk,
                                   kDenseGemmHalfBlockK);
#pragma unroll
            for (int c = 0; c < kDenseGemmWideAccCols; ++c) {
                const int tile_col = warp_n_group * kDenseGemmWideAccCols + c;
                wmma::load_matrix_sync(
                    b_frag, smem_b[stage] + tile_col * kDenseMmaN * kDenseGemmHalfBlockK + kk,
                    kDenseGemmHalfBlockK);
                wmma::mma_sync(c_frag[c], a_frag, b_frag, c_frag[c]);
            }
        }

        if (next_k0 < k) { dense_cp_async_wait_all(); }
        __syncthreads();
        stage = next_stage;
    }

#pragma unroll
    for (int tc = 0; tc < kDenseGemmWideAccCols; ++tc) {
        float* tile_c = smem_c[warp][tc];
        wmma::store_matrix_sync(tile_c, c_frag[tc], kDenseMmaM, wmma::mem_col_major);
        __syncwarp();

        for (int i = lane; i < kDenseMmaTileElements; i += 32) {
            const std::int32_t r                          = i & (kDenseMmaM - 1);
            const std::int32_t c                          = i >> 4;
            const std::int32_t row                        = warp_row0 + r;
            const std::int32_t col                        = warp_col0 + tc * kDenseMmaN + c;
            out[row + static_cast<std::int64_t>(n) * col] = __float2bfloat16(tile_c[i]);
        }
        __syncwarp();
    }
}

__global__ void linear_dense_gemm_warp_kernel(const __nv_bfloat16* x, const void* weight,
                                              __nv_bfloat16* out, std::int32_t n, std::int32_t k,
                                              std::int32_t t, bool weight_fp32) {
    using namespace nvcuda;

    __shared__ __align__(16) __nv_bfloat16 smem_a[kDenseGemmWarps][kDenseMmaTileElements];
    __shared__ __align__(16) __nv_bfloat16 smem_b[kDenseGemmWarps][kDenseMmaTileElements];
    __shared__ __align__(16) float smem_c[kDenseGemmWarps][kDenseMmaTileElements];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const std::int32_t tiles_n = (n + kDenseMmaM - 1) / kDenseMmaM;
    const std::int32_t tiles_t = (t + kDenseMmaN - 1) / kDenseMmaN;
    const std::int32_t tile_id = static_cast<std::int32_t>(blockIdx.x) * kDenseGemmWarps + warp;
    if (tile_id >= tiles_n * tiles_t) { return; }

    const std::int32_t tile_n = tile_id % tiles_n;
    const std::int32_t tile_t = tile_id / tiles_n;
    const std::int32_t row0   = tile_n * kDenseMmaM;
    const std::int32_t col0   = tile_t * kDenseMmaN;

    wmma::fragment<wmma::matrix_a, kDenseMmaM, kDenseMmaN, kDenseMmaK, __nv_bfloat16,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, kDenseMmaM, kDenseMmaN, kDenseMmaK, __nv_bfloat16,
                   wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, kDenseMmaM, kDenseMmaN, kDenseMmaK, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    __nv_bfloat16* tile_a = smem_a[warp];
    __nv_bfloat16* tile_b = smem_b[warp];

    if (!weight_fp32 && row0 + kDenseMmaM <= n && col0 + kDenseMmaN <= t && (k % kDenseMmaK) == 0) {
        const auto* w_bf16 = static_cast<const __nv_bfloat16*>(weight);
        for (std::int32_t k0 = 0; k0 < k; k0 += kDenseMmaK) {
            wmma::load_matrix_sync(a_frag, w_bf16 + static_cast<std::int64_t>(row0) * k + k0, k);
            wmma::load_matrix_sync(b_frag, x + k0 + static_cast<std::int64_t>(col0) * k, k);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    } else {
        for (std::int32_t k0 = 0; k0 < k; k0 += kDenseMmaK) {
            for (int i = lane; i < kDenseMmaTileElements; i += 32) {
                const std::int32_t r   = i / kDenseMmaK;
                const std::int32_t kk  = i - r * kDenseMmaK;
                const std::int32_t row = row0 + r;
                const std::int32_t col = k0 + kk;
                __nv_bfloat16 v        = __float2bfloat16(0.0f);
                if (row < n && col < k) {
                    const std::int64_t idx = static_cast<std::int64_t>(row) * k + col;
                    v = weight_fp32 ? __float2bfloat16(static_cast<const float*>(weight)[idx])
                                    : static_cast<const __nv_bfloat16*>(weight)[idx];
                }
                tile_a[i] = v;
            }
            for (int i = lane; i < kDenseMmaTileElements; i += 32) {
                const std::int32_t kk  = i & (kDenseMmaK - 1);
                const std::int32_t c   = i >> 4;
                const std::int32_t row = k0 + kk;
                const std::int32_t col = col0 + c;
                __nv_bfloat16 v        = __float2bfloat16(0.0f);
                if (row < k && col < t) { v = x[row + static_cast<std::int64_t>(k) * col]; }
                tile_b[i] = v;
            }
            __syncwarp();

            wmma::load_matrix_sync(a_frag, tile_a, kDenseMmaK);
            wmma::load_matrix_sync(b_frag, tile_b, kDenseMmaK);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncwarp();
        }
    }

    float* tile_c = smem_c[warp];
    wmma::store_matrix_sync(tile_c, c_frag, kDenseMmaM, wmma::mem_col_major);
    __syncwarp();

    for (int i = lane; i < kDenseMmaTileElements; i += 32) {
        const std::int32_t r   = i & (kDenseMmaM - 1);
        const std::int32_t c   = i >> 4;
        const std::int32_t row = row0 + r;
        const std::int32_t col = col0 + c;
        if (row < n && col < t) {
            out[row + static_cast<std::int64_t>(n) * col] = __float2bfloat16(tile_c[i]);
        }
    }
}

} // namespace qus::kernels
