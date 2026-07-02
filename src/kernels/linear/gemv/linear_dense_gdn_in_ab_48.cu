#include "kernels/linear/gemv/linear_dense_gdn_in_ab_48.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>
#include <mma.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace qus::kernels::detail {
namespace {
namespace wmma = nvcuda::wmma;

constexpr int kN       = 48;
constexpr int kK       = 5120;
constexpr int kThreads = 256;
constexpr int kMmaM    = 16;
constexpr int kMmaN    = 16;
constexpr int kMmaK    = 16;
constexpr int kPrefillWarpsPerBlock = 4;

template <int Width = 32>
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset, Width);
    }
    return v;
}

template <int BlockSize>
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[BlockSize / 32];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) { warp_sums[warp] = v; }
    __syncthreads();

    v = (static_cast<int>(threadIdx.x) < (BlockSize / 32)) ? warp_sums[lane] : 0.0f;
    if (warp == 0) { v = warp_reduce_sum<BlockSize / 32>(v); }
    return v;
}

__device__ __forceinline__ float2 bf162_to_float2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

__global__ void linear_dense_gdn_in_ab_48_kernel(const __nv_bfloat16* x,
                                                 const __nv_bfloat16* a_weight,
                                                 const __nv_bfloat16* b_weight,
                                                 __nv_bfloat16* a_out,
                                                 __nv_bfloat16* b_out) {
    const int global_row = static_cast<int>(blockIdx.x);
    const bool is_b      = global_row >= kN;
    const int row        = is_b ? global_row - kN : global_row;
    const auto* weight   = is_b ? b_weight : a_weight;

    float acc                   = 0.0f;
    constexpr int kPairs        = kK / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kK;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<std::int64_t>(row_base));
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
        const float2 wf = bf162_to_float2(w2[p]);
        const float2 xf = bf162_to_float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }

    acc = block_reduce_sum<kThreads>(acc);
    if (threadIdx.x == 0) {
        if (is_b) {
            b_out[row] = __float2bfloat16(acc);
        } else {
            a_out[row] = __float2bfloat16(acc);
        }
    }
}

__device__ __forceinline__ float gdn_ab_softplus_f32(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float gdn_ab_sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <int NFrags>
__global__ void linear_dense_gdn_in_ab_gated_prefill_48_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* a_weight, const __nv_bfloat16* b_weight,
    const float* A_log, const float* dt_bias, float* g, float* beta, std::int32_t t) {
    constexpr int kWarpN = NFrags * kMmaN;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int tile_row0 = static_cast<int>(blockIdx.y) * kMmaM;
    const int tile_col0 =
        (static_cast<int>(blockIdx.x) * kPrefillWarpsPerBlock + warp) * kWarpN;
    if (tile_col0 >= t) { return; }

    __shared__ __align__(16) __nv_bfloat16 b_tail[kPrefillWarpsPerBlock][NFrags]
                                                        [kMmaK * kMmaN];
    __shared__ __align__(16) float c_tile[kPrefillWarpsPerBlock][NFrags][2]
                                           [kMmaM * kMmaN];

    wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, __nv_bfloat16, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_a, kMmaM, kMmaN, kMmaK, __nv_bfloat16, wmma::row_major>
        b_weight_frag;
    wmma::fragment<wmma::matrix_b, kMmaM, kMmaN, kMmaK, __nv_bfloat16, wmma::col_major>
        x_frag[NFrags];
    wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> a_acc[NFrags];
    wmma::fragment<wmma::accumulator, kMmaM, kMmaN, kMmaK, float> b_acc[NFrags];
#pragma unroll
    for (int n = 0; n < NFrags; ++n) {
        wmma::fill_fragment(a_acc[n], 0.0f);
        wmma::fill_fragment(b_acc[n], 0.0f);
    }

    for (int k0 = 0; k0 < kK; k0 += kMmaK) {
        const __nv_bfloat16* a_ptr =
            a_weight + static_cast<std::int64_t>(tile_row0) * kK + k0;
        const __nv_bfloat16* bw_ptr =
            b_weight + static_cast<std::int64_t>(tile_row0) * kK + k0;
        wmma::load_matrix_sync(a_frag, a_ptr, kK);
        wmma::load_matrix_sync(b_weight_frag, bw_ptr, kK);

#pragma unroll
        for (int n = 0; n < NFrags; ++n) {
            const int col0 = tile_col0 + n * kMmaN;
            if (col0 + kMmaN <= t) {
                const __nv_bfloat16* x_ptr = x + static_cast<std::int64_t>(col0) * kK + k0;
                wmma::load_matrix_sync(x_frag[n], x_ptr, kK);
            } else {
                for (int i = lane; i < kMmaK * kMmaN; i += 32) {
                    const int kk    = i & (kMmaK - 1);
                    const int col   = i >> 4;
                    const int token = col0 + col;
                    b_tail[warp][n][i] =
                        (token < t) ? x[static_cast<std::int64_t>(token) * kK + k0 + kk]
                                    : __float2bfloat16(0.0f);
                }
                __syncwarp();
                wmma::load_matrix_sync(x_frag[n], b_tail[warp][n], kMmaK);
            }
            wmma::mma_sync(a_acc[n], a_frag, x_frag[n], a_acc[n]);
            wmma::mma_sync(b_acc[n], b_weight_frag, x_frag[n], b_acc[n]);
        }
    }

#pragma unroll
    for (int n = 0; n < NFrags; ++n) {
        wmma::store_matrix_sync(c_tile[warp][n][0], a_acc[n], kMmaN, wmma::mem_row_major);
        wmma::store_matrix_sync(c_tile[warp][n][1], b_acc[n], kMmaN, wmma::mem_row_major);
    }
    __syncwarp();

#pragma unroll
    for (int n = 0; n < NFrags; ++n) {
        for (int i = lane; i < kMmaM * kMmaN; i += 32) {
            const int row   = i >> 4;
            const int col   = i & (kMmaN - 1);
            const int token = tile_col0 + n * kMmaN + col;
            if (token >= t) { continue; }

            const int global_row = tile_row0 + row;
            const float a_rounded = __bfloat162float(__float2bfloat16(c_tile[warp][n][0][i]));
            const float b_rounded = __bfloat162float(__float2bfloat16(c_tile[warp][n][1][i]));
            const std::int64_t out_index = static_cast<std::int64_t>(token) * kN + global_row;
            const float sp = gdn_ab_softplus_f32(a_rounded + dt_bias[global_row]);
            g[out_index] = -expf(A_log[global_row]) * sp;
            beta[out_index] = gdn_ab_sigmoid_f32(b_rounded);
        }
    }
}

template <int NFrags>
void launch_gdn_in_ab_prefill(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                              Tensor& beta, cudaStream_t stream) {
    constexpr int kWarpN = NFrags * kMmaN;
    const std::int32_t t = x.ne[1];
    const int n_tiles = (t + kWarpN - 1) / kWarpN;
    dim3 block(kPrefillWarpsPerBlock * 32);
    dim3 grid((n_tiles + kPrefillWarpsPerBlock - 1) / kPrefillWarpsPerBlock, kN / kMmaM);
    linear_dense_gdn_in_ab_gated_prefill_48_kernel<NFrags><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), t);
}

__global__ void linear_dense_gdn_in_ab_gated_48_kernel(
    const __nv_bfloat16* x, const __nv_bfloat16* a_weight, const __nv_bfloat16* b_weight,
    const float* A_log, const float* dt_bias, float* g, float* beta) {
    const int global_row = static_cast<int>(blockIdx.x);
    const bool is_b      = global_row >= kN;
    const int row        = is_b ? global_row - kN : global_row;
    const auto* weight   = is_b ? b_weight : a_weight;

    float acc                   = 0.0f;
    constexpr int kPairs        = kK / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kK;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<std::int64_t>(row_base));
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
        const float2 wf = bf162_to_float2(w2[p]);
        const float2 xf = bf162_to_float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }

    acc = block_reduce_sum<kThreads>(acc);
    if (threadIdx.x == 0) {
        const float rounded = __bfloat162float(__float2bfloat16(acc));
        if (is_b) {
            beta[row] = gdn_ab_sigmoid_f32(rounded);
        } else {
            const float sp = gdn_ab_softplus_f32(rounded + dt_bias[row]);
            g[row] = -expf(A_log[row]) * sp;
        }
    }
}

void require_shape(const Weight& w, const char* name) {
    if (w.n != kN || w.k != kK || w.shape[0] != kN || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_in_ab_decode: ") + name +
                                    " requires 48x5120 dense BF16");
    }
}

} // namespace

void linear_dense_gdn_in_ab_48_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, Tensor& a_out, Tensor& b_out,
                                      cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    linear_dense_gdn_in_ab_48_kernel<<<2 * kN, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<__nv_bfloat16*>(a_out.data),
        static_cast<__nv_bfloat16*>(b_out.data));
    CUDA_CHECK(cudaGetLastError());
}

void linear_dense_gdn_in_ab_gated_48_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    linear_dense_gdn_in_ab_gated_48_kernel<<<2 * kN, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
}

void linear_dense_gdn_in_ab_gated_prefill_48_launch(const Tensor& x, const Weight& a_weight,
                                                    const Weight& b_weight, const Tensor& A_log,
                                                    const Tensor& dt_bias, Tensor& g,
                                                    Tensor& beta, cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    const std::int32_t t = x.ne[1];
    if (t == 0) { return; }
    if (t >= 8192) {
        launch_gdn_in_ab_prefill<2>(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
    } else {
        launch_gdn_in_ab_prefill<1>(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace qus::kernels::detail
