#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                  = 48;
constexpr int kK                  = 5120;
constexpr int kThreads            = 256;
constexpr int kLogicalRows        = 2 * kN;
constexpr int kSmallTMax          = 8;
constexpr int kSmallTKSlice       = 512;
constexpr int kSmallTSplits       = kK / kSmallTKSlice;
constexpr int kSmallTRowsPerBlock = 4;
constexpr int kSmallTThreads      = kSmallTRowsPerBlock * 32;
static_assert(kK % kSmallTKSlice == 0, "small-T K split must divide K");

constexpr int k35N           = 32;
constexpr int k35K           = 2048;
constexpr int k35LogicalRows = 2 * k35N;

template <int TokenTile, int KSlice, int RowsPerBlock>
__global__ void bf16_gdn_gating_proj_small_t_partial_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, float* __restrict__ partial, std::int32_t t) {
    static_assert(TokenTile == kSmallTMax, "small-T token tile is fixed to 8");
    static_assert(KSlice == kSmallTKSlice, "small-T K split is fixed to 512");
    static_assert(RowsPerBlock == kSmallTRowsPerBlock, "small-T rows/block mismatch");
    constexpr int kVecsPerCol = KSlice / 8;

    __shared__ __align__(16) __nv_bfloat16 x_sh[TokenTile][KSlice];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
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
            load_vec<uint4>(x + static_cast<std::int64_t>(token0 + col) * kK + k0 + vec * 8);
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
        const uint4 wv   = load_vec<uint4>(wrow + vec * 8);
        const float2 wf0 = bf16x2_bits_to_float2(wv.x);
        const float2 wf1 = bf16x2_bits_to_float2(wv.y);
        const float2 wf2 = bf16x2_bits_to_float2(wv.z);
        const float2 wf3 = bf16x2_bits_to_float2(wv.w);

#pragma unroll
        for (int tt = 0; tt < TokenTile; ++tt) {
            if (tt < ncols) {
                const uint4 xv   = x_sh_v[tt * kVecsPerCol + vec];
                const float2 xf0 = bf16x2_bits_to_float2(xv.x);
                const float2 xf1 = bf16x2_bits_to_float2(xv.y);
                const float2 xf2 = bf16x2_bits_to_float2(xv.z);
                const float2 xf3 = bf16x2_bits_to_float2(xv.w);
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
                const int token      = token0 + tt;
                partial[(static_cast<std::int64_t>(split) * t + token) * kLogicalRows +
                        logical_row] = sum;
            }
        }
    }
}

__global__ void bf16_gdn_gating_proj_small_t_reduce_kernel(const float* __restrict__ partial,
                                                           const float* __restrict__ A_log,
                                                           const float* __restrict__ dt_bias,
                                                           float* __restrict__ g,
                                                           float* __restrict__ beta,
                                                           std::int32_t t) {
    const int i =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int elems = kN * t;
    if (i >= elems) { return; }

    const int row   = i % kN;
    const int token = i / kN;
    float acc_a     = 0.0f;
    float acc_b     = 0.0f;
#pragma unroll
    for (int split = 0; split < kSmallTSplits; ++split) {
        const std::int64_t base = (static_cast<std::int64_t>(split) * t + token) * kLogicalRows;
        acc_a += partial[base + row];
        acc_b += partial[base + kN + row];
    }

    const std::int64_t out_index = static_cast<std::int64_t>(token) * kN + row;
    const float sp               = softplus(acc_a + dt_bias[row]);
    g[out_index]                 = -expf(A_log[row]) * sp;
    beta[out_index]              = sigmoid(acc_b);
}

__global__ void bf16_gdn_gating_proj_gemv_kernel(const __nv_bfloat16* x,
                                                 const __nv_bfloat16* a_weight,
                                                 const __nv_bfloat16* b_weight, const float* A_log,
                                                 const float* dt_bias, float* g, float* beta) {
    const int global_row = static_cast<int>(blockIdx.x);
    const bool is_b      = global_row >= kN;
    const int row        = is_b ? global_row - kN : global_row;
    const auto* weight   = is_b ? b_weight : a_weight;
    __shared__ float warp_sums[kThreads / kWarpSize];

    float acc                   = 0.0f;
    constexpr int kPairs        = kK / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kK;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<std::int64_t>(row_base));
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
        const float2 wf = bf16x2_to_float2(w2[p]);
        const float2 xf = bf16x2_to_float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }

    acc = block_reduce_sum<kThreads>(acc, warp_sums);
    if (threadIdx.x == 0) {
        if (is_b) {
            beta[row] = sigmoid(acc);
        } else {
            const float sp = softplus(acc + dt_bias[row]);
            g[row]         = -expf(A_log[row]) * sp;
        }
    }
}

template <int ColsPerTile>
__global__ void bf16_gdn_gating_proj_35_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                                    const __nv_bfloat16* __restrict__ a_weight,
                                                    const __nv_bfloat16* __restrict__ b_weight,
                                                    const float* __restrict__ A_log,
                                                    const float* __restrict__ dt_bias,
                                                    float* __restrict__ g, float* __restrict__ beta,
                                                    std::int32_t t) {
    static_assert(ColsPerTile == 4 || ColsPerTile == 8);
    const int lane        = static_cast<int>(threadIdx.x);
    const int logical_row = static_cast<int>(blockIdx.x);
    const bool is_b       = logical_row >= k35N;
    const int row         = is_b ? logical_row - k35N : logical_row;
    const int col0        = static_cast<int>(blockIdx.y) * ColsPerTile;
    const int ncols       = min(ColsPerTile, t - col0);
    const auto* weight    = is_b ? b_weight : a_weight;
    const auto* wrow      = weight + static_cast<std::int64_t>(row) * k35K;

    float acc[ColsPerTile];
#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) { acc[col] = 0.0f; }

    constexpr int kVecs = k35K / 8;
    for (int vec = lane; vec < kVecs; vec += 32) {
        const uint4 wv   = load_vec<uint4>(wrow + vec * 8);
        const float2 wf0 = bf16x2_bits_to_float2(wv.x);
        const float2 wf1 = bf16x2_bits_to_float2(wv.y);
        const float2 wf2 = bf16x2_bits_to_float2(wv.z);
        const float2 wf3 = bf16x2_bits_to_float2(wv.w);
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            if (col < ncols) {
                const uint4 xv =
                    load_vec<uint4>(x + static_cast<std::int64_t>(col0 + col) * k35K + vec * 8);
                const float2 xf0 = bf16x2_bits_to_float2(xv.x);
                const float2 xf1 = bf16x2_bits_to_float2(xv.y);
                const float2 xf2 = bf16x2_bits_to_float2(xv.z);
                const float2 xf3 = bf16x2_bits_to_float2(xv.w);
                acc[col]         = fmaf(wf0.x, xf0.x, acc[col]);
                acc[col]         = fmaf(wf0.y, xf0.y, acc[col]);
                acc[col]         = fmaf(wf1.x, xf1.x, acc[col]);
                acc[col]         = fmaf(wf1.y, xf1.y, acc[col]);
                acc[col]         = fmaf(wf2.x, xf2.x, acc[col]);
                acc[col]         = fmaf(wf2.y, xf2.y, acc[col]);
                acc[col]         = fmaf(wf3.x, xf3.x, acc[col]);
                acc[col]         = fmaf(wf3.y, xf3.y, acc[col]);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) {
        if (col < ncols) {
            const float sum = warp_reduce_sum(acc[col]);
            if (lane == 0) {
                const std::int64_t out_index = static_cast<std::int64_t>(col0 + col) * k35N + row;
                if (is_b) {
                    beta[out_index] = sigmoid(sum);
                } else {
                    g[out_index] = -expf(A_log[row]) * softplus(sum + dt_bias[row]);
                }
            }
        }
    }
}

void require_shape(const Weight& w, const char* name) {
    if (w.n != kN || w.k != kK || w.shape[0] != kN || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_gating_proj: ") + name +
                                    " requires contiguous BF16 [48,5120]");
    }
}

void require_shape35(const Weight& w, const char* name) {
    if (w.n != k35N || w.k != k35K || w.shape[0] != k35N || w.shape[1] != k35K) {
        throw std::invalid_argument(std::string("gdn_gating_proj: ") + name +
                                    " requires contiguous BF16 [32,2048]");
    }
}

template <class Geometry, int SplitK, int Warps = kBf16GdnWarps, bool NormalizeInput = false>
void launch_bf16_prefill_mma(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                             const Tensor* norm_weight, float norm_eps, Tensor* normalized_x,
                             const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                             const Tensor& dt_bias, void* workspace, Tensor& g, Tensor& beta,
                             cudaStream_t stream) {
    const std::int32_t t     = x.ne[1];
    constexpr int kBlockN    = Geometry::kBlockN;
    constexpr int kSmemBytes = kBf16GdnSmemBytes<kBlockN>;
    const dim3 block(Warps * 32);
    const dim3 grid(static_cast<unsigned>(div_up(t, kBlockN)),
                    static_cast<unsigned>(Geometry::kHeads / kBf16GdnBlockM),
                    static_cast<unsigned>(SplitK));
    auto launch = [&](auto full_tokens) {
        constexpr bool FullTokens = decltype(full_tokens)::value;
        static const cudaError_t attr =
            cudaFuncSetAttribute(bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens,
                                                                      Warps, NormalizeInput>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        CUDA_CHECK(attr);
        if constexpr (SplitK > 1) {
            cudaLaunchConfig_t config{};
            config.gridDim          = grid;
            config.blockDim         = block;
            config.dynamicSmemBytes = kSmemBytes;
            config.stream           = stream;
            cudaLaunchAttribute cooperative{};
            cooperative.id              = cudaLaunchAttributeCooperative;
            cooperative.val.cooperative = 1;
            config.attrs                = &cooperative;
            config.numAttrs             = 1;
            CUDA_CHECK(cudaLaunchKernelEx(
                &config,
                bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                                                     NormalizeInput>,
                static_cast<const __nv_bfloat16*>(x.data),
                norm_weight != nullptr ? static_cast<const __nv_bfloat16*>(norm_weight->data)
                                       : static_cast<const __nv_bfloat16*>(nullptr),
                normalized_x != nullptr ? static_cast<__nv_bfloat16*>(normalized_x->data)
                                        : static_cast<__nv_bfloat16*>(nullptr),
                norm_eps, static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata),
                static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                static_cast<float*>(workspace), static_cast<float*>(g.data),
                static_cast<float*>(beta.data), t));
        } else {
            bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                                                 NormalizeInput>
                <<<grid, block, kSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    norm_weight != nullptr ? static_cast<const __nv_bfloat16*>(norm_weight->data)
                                           : static_cast<const __nv_bfloat16*>(nullptr),
                    normalized_x != nullptr ? static_cast<__nv_bfloat16*>(normalized_x->data)
                                            : static_cast<__nv_bfloat16*>(nullptr),
                    norm_eps, static_cast<const __nv_bfloat16*>(a_weight.qdata),
                    static_cast<const __nv_bfloat16*>(b_weight.qdata),
                    static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                    static_cast<float*>(workspace), static_cast<float*>(g.data),
                    static_cast<float*>(beta.data), t);
        }
    };
    if (variant == Bf16GdnGatingTokenVariant::Full) {
        launch(std::true_type{});
    } else if (variant == Bf16GdnGatingTokenVariant::Predicated) {
        launch(std::false_type{});
    } else {
        throw std::invalid_argument(
            "BF16 GDN gating MMA requires Full or Predicated token variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    bf16_gdn_gating_proj_gemv_kernel<<<2 * kN, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    const std::int32_t t       = x.ne[1];
    const std::size_t required = static_cast<std::size_t>(kSmallTSplits) *
                                 static_cast<std::size_t>(t) *
                                 static_cast<std::size_t>(kLogicalRows) * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
        throw std::invalid_argument("gdn_gating_proj: small-T workspace is too small");
    }

    dim3 partial_block(kSmallTThreads);
    dim3 partial_grid(div_up(kLogicalRows, kSmallTRowsPerBlock), kSmallTSplits,
                      div_up(t, kSmallTMax));
    bf16_gdn_gating_proj_small_t_partial_kernel<kSmallTMax, kSmallTKSlice, kSmallTRowsPerBlock>
        <<<partial_grid, partial_block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<float*>(workspace), t);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kReduceThreads = 128;
    const int reduce_elems       = kN * t;
    const int reduce_blocks      = div_up(reduce_elems, kReduceThreads);
    bf16_gdn_gating_proj_small_t_reduce_kernel<<<reduce_blocks, kReduceThreads, 0, stream>>>(
        static_cast<const float*>(workspace), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), t);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 8, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, workspace, g, beta,
                                                     stream);
}

void bf16_gdn_gating_proj_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 4>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                  b_weight, A_log, dt_bias, workspace, g, beta,
                                                  stream);
}

void bf16_gdn_gating_proj_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 2>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                  b_weight, A_log, dt_bias, workspace, g, beta,
                                                  stream);
}

void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 1, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, nullptr, g, beta,
                                                     stream);
}

template <int ColsPerTile>
void launch_35_simt(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                    const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta,
                    cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    const dim3 grid(static_cast<unsigned>(k35LogicalRows),
                    static_cast<unsigned>(div_up(x.ne[1], ColsPerTile)), 1u);
    bf16_gdn_gating_proj_35_simt_kernel<ColsPerTile><<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_35_simt_c4_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_35_simt<4>(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
}

void bf16_gdn_gating_proj_35_simt_c8_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_35_simt<8>(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
}

void bf16_gdn_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 32, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                      b_weight, A_log, dt_bias, workspace, g, beta,
                                                      stream);
}

void bf16_gdn_norm_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant,
                                                     const Tensor& x, const Tensor& norm_weight,
                                                     float eps, Tensor& h, const Weight& a_weight,
                                                     const Weight& b_weight, const Tensor& A_log,
                                                     const Tensor& dt_bias, void* workspace,
                                                     Tensor& g, Tensor& beta, cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 32, 8, true>(variant, x, &norm_weight, eps, &h,
                                                            a_weight, b_weight, A_log, dt_bias,
                                                            workspace, g, beta, stream);
}

void bf16_gdn_gating_proj_35_mma_split16_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 16, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                      b_weight, A_log, dt_bias, workspace, g, beta,
                                                      stream);
}

void bf16_gdn_gating_proj_35_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 8, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, workspace, g, beta,
                                                     stream);
}

void bf16_gdn_gating_proj_35_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 4, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, workspace, g, beta,
                                                     stream);
}

void bf16_gdn_gating_proj_35_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 2, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, workspace, g, beta,
                                                     stream);
}

void bf16_gdn_gating_proj_35_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                Tensor& g, Tensor& beta, cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 1, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, nullptr, g, beta,
                                                     stream);
}

} // namespace ninfer::ops::detail
