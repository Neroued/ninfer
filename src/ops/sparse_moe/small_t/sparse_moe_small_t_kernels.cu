#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/sparse_moe/decode/sparse_moe_decode.h"
#include "ops/sparse_moe/sparse_moe_route.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden           = 2048;
constexpr int kRouterRows       = 257;
constexpr int kTopK             = 8;
constexpr int kRouterPartitions = 4;
constexpr int kRouterWarps      = 4;

template <int Tokens>
__global__ void sparse_moe_small_t_s1_kernel(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ router,
                                             float* __restrict__ partial_scores) {
    static_assert(Tokens >= 1 && Tokens <= kSparseMoeSmallTMax);
    __shared__ float partial[kRouterWarps][Tokens];
    const int row       = static_cast<int>(blockIdx.x) / kRouterPartitions;
    const int partition = static_cast<int>(blockIdx.x) - row * kRouterPartitions;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int k = partition * (kHidden / kRouterPartitions) + static_cast<int>(threadIdx.x) * 4;

    const uint2 weight = load_vec<uint2>(router + static_cast<std::int64_t>(row) * kHidden + k);
    const float2 w0    = bf16x2_bits_to_float2(weight.x);
    const float2 w1    = bf16x2_bits_to_float2(weight.y);

    float accumulator[Tokens];
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const uint2 input  = load_vec<uint2>(x + static_cast<std::int64_t>(token) * kHidden + k);
        const float2 x0    = bf16x2_bits_to_float2(input.x);
        const float2 x1    = bf16x2_bits_to_float2(input.y);
        float sum          = 0.0f;
        sum                = fmaf(w0.x, x0.x, sum);
        sum                = fmaf(w0.y, x0.y, sum);
        sum                = fmaf(w1.x, x1.x, sum);
        accumulator[token] = fmaf(w1.y, x1.y, sum);
    }

#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float sum = warp_reduce_sum(accumulator[token]);
        if (lane == 0) { partial[warp][token] = sum; }
    }
    __syncthreads();

    if (warp == 0) {
        for (int token = lane; token < Tokens; token += 32) {
            float sum = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < kRouterWarps; ++source_warp) {
                sum += partial[source_warp][token];
            }
            partial_scores[(static_cast<std::int64_t>(token) * kRouterRows + row) *
                               kRouterPartitions +
                           partition] = sum;
        }
    }
}

template <int Tokens>
__global__ void
sparse_moe_small_t_s2_kernel(const float* __restrict__ partial_scores, int* __restrict__ token_ids,
                             float* __restrict__ token_alpha, float* __restrict__ shared_scale) {
    static_assert(Tokens >= 1 && Tokens <= kSparseMoeSmallTMax);
    __shared__ float scores[Tokens][kRouterRows];
    __shared__ float selected_logits[Tokens][kTopK];
    const int tid = static_cast<int>(threadIdx.x);
    for (int index = tid; index < Tokens * kRouterRows; index += static_cast<int>(blockDim.x)) {
        float sum = 0.0f;
#pragma unroll
        for (int partition = 0; partition < kRouterPartitions; ++partition) {
            sum += partial_scores[static_cast<std::int64_t>(index) * kRouterPartitions + partition];
        }
        reinterpret_cast<float*>(scores)[index] = sum;
    }
    __syncthreads();

    const int resident_warps = static_cast<int>(blockDim.x) >> 5;
    for (int token = tid >> 5; token < Tokens; token += resident_warps) {
        sparse_moe_select_top8_warp(scores[token], token_ids + token * kTopK,
                                    token_alpha + token * kTopK, shared_scale + token,
                                    selected_logits[token]);
    }
}

template <int Tokens>
void launch_s1(const __nv_bfloat16* x, const __nv_bfloat16* router, float* partial_scores,
               cudaStream_t stream) {
    sparse_moe_small_t_s1_kernel<Tokens>
        <<<kRouterRows * kRouterPartitions, kRouterWarps * 32, 0, stream>>>(
            x, router, partial_scores);
    CUDA_CHECK(cudaGetLastError());
}

template <int Tokens>
void launch_s2(const float* partial_scores, int* token_ids, float* token_alpha, float* shared_scale,
               cudaStream_t stream) {
    sparse_moe_small_t_s2_kernel<Tokens>
        <<<1, (Tokens < 32 ? Tokens : 32) * 32, 0, stream>>>(partial_scores, token_ids, token_alpha,
                                                             shared_scale);
    CUDA_CHECK(cudaGetLastError());
}

void launch_s3_tiled(const Tensor& x, const SparseMoeWeights& weights,
                     const SparseMoeSmallTPlan& plan, const SparseMoeSmallTWorkspace& workspace,
                     cudaStream_t stream) {
    sparse_moe_decode_launch_d3_small_t(
        x, weights, static_cast<const int*>(workspace.token_ids.data),
        static_cast<float*>(workspace.scratch.data), plan.tokens, stream);
}

void launch_s4_tiled(const SparseMoeWeights& weights, Tensor& destination,
                     const SparseMoeSmallTPlan& plan, const SparseMoeSmallTWorkspace& workspace,
                     cudaStream_t stream) {
    sparse_moe_decode_launch_d4_small_t(
        weights, destination, static_cast<const int*>(workspace.token_ids.data),
        static_cast<const float*>(workspace.token_alpha.data),
        static_cast<const float*>(workspace.shared_scale.data),
        static_cast<const float*>(workspace.scratch.data), plan.tokens, stream);
}

template <class Launch>
void dispatch_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
    case 1:
        launch.template operator()<1>();
        return;
    case 2:
        launch.template operator()<2>();
        return;
    case 3:
        launch.template operator()<3>();
        return;
    case 4:
        launch.template operator()<4>();
        return;
    case 5:
        launch.template operator()<5>();
        return;
    case 6:
        launch.template operator()<6>();
        return;
    case 7:
        launch.template operator()<7>();
        return;
    case 8:
        launch.template operator()<8>();
        return;
    case 9:
        launch.template operator()<9>();
        return;
    case 10:
        launch.template operator()<10>();
        return;
    case 11:
        launch.template operator()<11>();
        return;
    case 12:
        launch.template operator()<12>();
        return;
    case 13:
        launch.template operator()<13>();
        return;
    case 14:
        launch.template operator()<14>();
        return;
    case 15:
        launch.template operator()<15>();
        return;
    case 16:
        launch.template operator()<16>();
        return;
    case 17:
        launch.template operator()<17>();
        return;
    case 18:
        launch.template operator()<18>();
        return;
    case 19:
        launch.template operator()<19>();
        return;
    case 20:
        launch.template operator()<20>();
        return;
    case 21:
        launch.template operator()<21>();
        return;
    case 22:
        launch.template operator()<22>();
        return;
    case 23:
        launch.template operator()<23>();
        return;
    case 24:
        launch.template operator()<24>();
        return;
    case 25:
        launch.template operator()<25>();
        return;
    case 26:
        launch.template operator()<26>();
        return;
    case 27:
        launch.template operator()<27>();
        return;
    case 28:
        launch.template operator()<28>();
        return;
    case 29:
        launch.template operator()<29>();
        return;
    case 30:
        launch.template operator()<30>();
        return;
    case 31:
        launch.template operator()<31>();
        return;
    case 32:
        launch.template operator()<32>();
        return;
    case 33:
        launch.template operator()<33>();
        return;
    case 34:
        launch.template operator()<34>();
        return;
    case 35:
        launch.template operator()<35>();
        return;
    case 36:
        launch.template operator()<36>();
        return;
    case 37:
        launch.template operator()<37>();
        return;
    case 38:
        launch.template operator()<38>();
        return;
    case 39:
        launch.template operator()<39>();
        return;
    case 40:
        launch.template operator()<40>();
        return;
    case 41:
        launch.template operator()<41>();
        return;
    case 42:
        launch.template operator()<42>();
        return;
    case 43:
        launch.template operator()<43>();
        return;
    case 44:
        launch.template operator()<44>();
        return;
    default:
        throw std::invalid_argument("sparse_moe small-T: unsupported token count");
    }
}

} // namespace

void sparse_moe_small_t_launch_s1(const Tensor& x, const Weight& router_shared_gate,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    dispatch_tokens(x.ne[1], [&]<int Tokens>() {
        launch_s1<Tokens>(static_cast<const __nv_bfloat16*>(x.data),
                          static_cast<const __nv_bfloat16*>(router_shared_gate.qdata),
                          static_cast<float*>(workspace.scratch.data), stream);
    });
}

void sparse_moe_small_t_launch_s2(const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    dispatch_tokens(plan.tokens, [&]<int Tokens>() {
        launch_s2<Tokens>(static_cast<const float*>(workspace.scratch.data),
                          static_cast<int*>(workspace.token_ids.data),
                          static_cast<float*>(workspace.token_alpha.data),
                          static_cast<float*>(workspace.shared_scale.data), stream);
    });
}

void sparse_moe_small_t_launch_s3(const Tensor& x, const SparseMoeWeights& weights,
                                  const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    launch_s3_tiled(x, weights, plan, workspace, stream);
}

void sparse_moe_small_t_launch_s4(const SparseMoeWeights& weights, Tensor& destination,
                                  const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    launch_s4_tiled(weights, destination, plan, workspace, stream);
}

void sparse_moe_small_t_launch(const Tensor& x, const SparseMoeWeights& weights,
                               Tensor& destination, const SparseMoeSmallTPlan& plan,
                               const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    sparse_moe_small_t_launch_s1(x, weights.router_shared_gate, workspace, stream);
    sparse_moe_small_t_launch_s2(plan, workspace, stream);
    sparse_moe_small_t_launch_s3(x, weights, plan, workspace, stream);
    sparse_moe_small_t_launch_s4(weights, destination, plan, workspace, stream);
}

} // namespace ninfer::ops::detail
