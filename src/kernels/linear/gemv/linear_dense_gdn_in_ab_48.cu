#include "kernels/linear/gemv/linear_dense_gdn_in_ab_48.cuh"

#include "kernels/linear/gemm/linear_dense_gdn_in_ab_gemm_mma.cuh"

#include "qus/core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace qus::kernels::detail {
namespace {

constexpr int kN       = 48;
constexpr int kK       = 5120;
constexpr int kThreads = 256;
constexpr int kLogicalRows = 2 * kN;
constexpr int kSmallTMax            = 8;
constexpr int kSmallTKSlice         = 512;
constexpr int kSmallTSplits         = kK / kSmallTKSlice;
constexpr int kSmallTRowsPerBlock   = 4;
constexpr int kSmallTThreads        = kSmallTRowsPerBlock * 32;
static_assert(kK % kSmallTKSlice == 0, "small-T K split must divide K");

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

__device__ __forceinline__ float2 bf162_from_u32(std::uint32_t v) {
    return __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&v));
}

template <int TokenTile, int KSlice, int RowsPerBlock>
__global__ void linear_dense_gdn_in_ab_gated_smallt_partial_48_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, float* __restrict__ partial, std::int32_t t) {
    static_assert(TokenTile == kSmallTMax, "small-T token tile is fixed to 8");
    static_assert(KSlice == kSmallTKSlice, "small-T K split is fixed to 512");
    static_assert(RowsPerBlock == kSmallTRowsPerBlock, "small-T rows/block mismatch");
    constexpr int kVecsPerCol = KSlice / 8;

    __shared__ __align__(16) __nv_bfloat16 x_sh[TokenTile][KSlice];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int token0 = static_cast<int>(blockIdx.z) * TokenTile;
    if (token0 >= t) { return; }
    const int ncols = min(TokenTile, t - token0);
    const int split = static_cast<int>(blockIdx.y);
    const int k0    = split * KSlice;

    auto* x_sh_v = reinterpret_cast<uint4*>(x_sh);
    for (int i = static_cast<int>(threadIdx.x); i < ncols * kVecsPerCol;
         i += static_cast<int>(blockDim.x)) {
        const int col = i / kVecsPerCol;
        const int vec = i - col * kVecsPerCol;
        x_sh_v[col * kVecsPerCol + vec] =
            *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token0 + col) * kK +
                                            k0 + vec * 8);
    }
    __syncthreads();

    const int logical_row = static_cast<int>(blockIdx.x) * RowsPerBlock + warp;
    if (logical_row >= kLogicalRows) { return; }
    const bool is_b = logical_row >= kN;
    const int row   = is_b ? logical_row - kN : logical_row;
    const __nv_bfloat16* wrow =
        (is_b ? b_weight : a_weight) + static_cast<std::int64_t>(row) * kK + k0;

    float acc[TokenTile];
#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) { acc[tt] = 0.0f; }

    for (int vec = lane; vec < kVecsPerCol; vec += 32) {
        const uint4 wv = *reinterpret_cast<const uint4*>(wrow + vec * 8);
        const float2 wf0 = bf162_from_u32(wv.x);
        const float2 wf1 = bf162_from_u32(wv.y);
        const float2 wf2 = bf162_from_u32(wv.z);
        const float2 wf3 = bf162_from_u32(wv.w);

#pragma unroll
        for (int tt = 0; tt < TokenTile; ++tt) {
            if (tt < ncols) {
                const uint4 xv = x_sh_v[tt * kVecsPerCol + vec];
                const float2 xf0 = bf162_from_u32(xv.x);
                const float2 xf1 = bf162_from_u32(xv.y);
                const float2 xf2 = bf162_from_u32(xv.z);
                const float2 xf3 = bf162_from_u32(xv.w);
                acc[tt]          = fmaf(wf0.x, xf0.x, acc[tt]);
                acc[tt]          = fmaf(wf0.y, xf0.y, acc[tt]);
                acc[tt]          = fmaf(wf1.x, xf1.x, acc[tt]);
                acc[tt]          = fmaf(wf1.y, xf1.y, acc[tt]);
                acc[tt]          = fmaf(wf2.x, xf2.x, acc[tt]);
                acc[tt]          = fmaf(wf2.y, xf2.y, acc[tt]);
                acc[tt]          = fmaf(wf3.x, xf3.x, acc[tt]);
                acc[tt]          = fmaf(wf3.y, xf3.y, acc[tt]);
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) {
        if (tt < ncols) {
            float sum = warp_reduce_sum(acc[tt]);
            if (lane == 0) {
                const int token = token0 + tt;
                partial[(static_cast<std::int64_t>(split) * t + token) * kLogicalRows +
                        logical_row] = sum;
            }
        }
    }
}

__device__ __forceinline__ float gdn_ab_softplus_f32(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}

__device__ __forceinline__ float gdn_ab_sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void linear_dense_gdn_in_ab_gated_smallt_reduce_48_kernel(
    const float* __restrict__ partial, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta,
    std::int32_t t) {
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    const int elems = kN * t;
    if (i >= elems) { return; }

    const int row   = i % kN;
    const int token = i / kN;
    float acc_a = 0.0f;
    float acc_b = 0.0f;
#pragma unroll
    for (int split = 0; split < kSmallTSplits; ++split) {
        const std::int64_t base =
            (static_cast<std::int64_t>(split) * t + token) * kLogicalRows;
        acc_a += partial[base + row];
        acc_b += partial[base + kN + row];
    }

    const float a_rounded = __bfloat162float(__float2bfloat16(acc_a));
    const float b_rounded = __bfloat162float(__float2bfloat16(acc_b));
    const std::int64_t out_index = static_cast<std::int64_t>(token) * kN + row;
    const float sp = gdn_ab_softplus_f32(a_rounded + dt_bias[row]);
    g[out_index] = -expf(A_log[row]) * sp;
    beta[out_index] = gdn_ab_sigmoid_f32(b_rounded);
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
        throw std::invalid_argument(std::string("gdn_in_ab_gated: ") + name +
                                    " requires 48x5120 dense BF16");
    }
}

int dense_prefill_split_k(std::int32_t tokens) {
    // Keep one K-association for every normal prefill chunk through 1024.
    // Changing SplitK changes FP32 grouping before the final BF16 round and can
    // make chunked and unchunked model execution cross an argmax boundary.
    if (tokens <= 1024) { return 8; }
    if (tokens <= 2048) { return 4; }
    if (tokens <= 4096) { return 2; }
    return 1;
}

template <int SplitK, int Warps = kDenseGdnWarps>
void launch_dense_prefill_mma(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, void* workspace,
                              Tensor& g, Tensor& beta, cudaStream_t stream) {
    const std::int32_t t = x.ne[1];
    const dim3 block(Warps * 32);
    const dim3 grid(static_cast<unsigned>((t + kDenseGdnBlockN - 1) / kDenseGdnBlockN),
                    static_cast<unsigned>(kDenseGdnHeads / kDenseGdnBlockM),
                    static_cast<unsigned>(SplitK));
    auto launch = [&](auto full_tokens) {
        constexpr bool FullTokens = decltype(full_tokens)::value;
        static const cudaError_t attr = cudaFuncSetAttribute(
            linear_dense_gdn_in_ab_gemm_mma_kernel<SplitK, FullTokens, Warps>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kDenseGdnSmemBytes);
        CUDA_CHECK(attr);
        if constexpr (SplitK > 1) {
            cudaLaunchConfig_t config{};
            config.gridDim          = grid;
            config.blockDim         = block;
            config.dynamicSmemBytes = kDenseGdnSmemBytes;
            config.stream           = stream;
            cudaLaunchAttribute cooperative{};
            cooperative.id              = cudaLaunchAttributeCooperative;
            cooperative.val.cooperative = 1;
            config.attrs                 = &cooperative;
            config.numAttrs              = 1;
            CUDA_CHECK(cudaLaunchKernelEx(
                &config, linear_dense_gdn_in_ab_gemm_mma_kernel<SplitK, FullTokens, Warps>,
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata),
                static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                static_cast<float*>(workspace), static_cast<float*>(g.data),
                static_cast<float*>(beta.data), t));
        } else {
            linear_dense_gdn_in_ab_gemm_mma_kernel<SplitK, FullTokens, Warps>
                <<<grid, block, kDenseGdnSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const __nv_bfloat16*>(a_weight.qdata),
                    static_cast<const __nv_bfloat16*>(b_weight.qdata),
                    static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                    static_cast<float*>(workspace), static_cast<float*>(g.data),
                    static_cast<float*>(beta.data), t);
        }
    };
    if ((t % kDenseGdnBlockN) == 0) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());

}

} // namespace

std::size_t linear_dense_gdn_in_ab_gated_48_workspace_bytes(std::int32_t tokens) {
    if (tokens >= 2 && tokens <= kSmallTMax) {
        return static_cast<std::size_t>(kSmallTSplits) * static_cast<std::size_t>(tokens) *
               static_cast<std::size_t>(kLogicalRows) * sizeof(float);
    }
    if (tokens > kSmallTMax) {
        const int split_k = dense_prefill_split_k(tokens);
        if (split_k > 1) {
            return static_cast<std::size_t>(split_k) * static_cast<std::size_t>(tokens) *
                   static_cast<std::size_t>(kLogicalRows) * sizeof(float);
        }
    }
    return 0;
}

void linear_dense_gdn_in_ab_gated_48_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, void* workspace,
                                            std::size_t workspace_bytes, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    const std::int32_t t = x.ne[1];
    if (t == 1) {
        linear_dense_gdn_in_ab_gated_48_kernel<<<2 * kN, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
            static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
            static_cast<float*>(beta.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    if (t <= kSmallTMax) {
        const std::size_t required = linear_dense_gdn_in_ab_gated_48_workspace_bytes(t);
        if (workspace == nullptr || workspace_bytes < required) {
            throw std::invalid_argument("gdn_in_ab_gated: small-T workspace is too small");
        }
        dim3 partial_block(kSmallTThreads);
        dim3 partial_grid((kLogicalRows + kSmallTRowsPerBlock - 1) / kSmallTRowsPerBlock,
                          kSmallTSplits, (t + kSmallTMax - 1) / kSmallTMax);
        linear_dense_gdn_in_ab_gated_smallt_partial_48_kernel<kSmallTMax, kSmallTKSlice,
                                                              kSmallTRowsPerBlock>
            <<<partial_grid, partial_block, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<float*>(workspace),
                t);
        CUDA_CHECK(cudaGetLastError());

        constexpr int kReduceThreads = 128;
        const int reduce_elems = kN * t;
        const int reduce_blocks = (reduce_elems + kReduceThreads - 1) / kReduceThreads;
        linear_dense_gdn_in_ab_gated_smallt_reduce_48_kernel<<<reduce_blocks, kReduceThreads, 0,
                                                               stream>>>(
            static_cast<const float*>(workspace), static_cast<const float*>(A_log.data),
            static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
            static_cast<float*>(beta.data), t);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const std::size_t required = linear_dense_gdn_in_ab_gated_48_workspace_bytes(t);
    if (required > 0 && (workspace == nullptr || workspace_bytes < required)) {
        throw std::invalid_argument("gdn_in_ab_gated: dense prefill workspace is too small");
    }
    switch (dense_prefill_split_k(t)) {
    case 8:
        launch_dense_prefill_mma<8, 8>(x, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
                                       stream);
        break;
    case 4:
        launch_dense_prefill_mma<4>(x, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
                                    stream);
        break;
    case 2:
        launch_dense_prefill_mma<2>(x, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
                                    stream);
        break;
    default:
        launch_dense_prefill_mma<1, 8>(x, a_weight, b_weight, A_log, dt_bias, workspace, g, beta,
                                       stream);
        break;
    }
}

} // namespace qus::kernels::detail
