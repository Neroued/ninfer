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
constexpr int kIntermediate     = 512;
constexpr int kPaths            = kTopK + 1;
constexpr int kRouterPartitions = 4;
constexpr int kRouterWarps      = 4;

template <int Tokens>
__global__ void sparse_moe_small_t_s1_kernel(const __nv_bfloat16* __restrict__ x,
                                             const __nv_bfloat16* __restrict__ router,
                                             float* __restrict__ partial_scores) {
    static_assert(Tokens >= kSparseMoeSmallTMin && Tokens <= kSparseMoeSmallTMax);
    __shared__ float partial[Tokens][kRouterWarps];
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
        if (lane == 0) { partial[token][warp] = sum; }
    }
    __syncthreads();

    if (warp == 0 && lane < Tokens) {
        float sum = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < kRouterWarps; ++source_warp) {
            sum += partial[lane][source_warp];
        }
        partial_scores[(static_cast<std::int64_t>(lane) * kRouterRows + row) * kRouterPartitions +
                       partition] = sum;
    }
}

template <int Tokens>
__global__ void
sparse_moe_small_t_s2_kernel(const float* __restrict__ partial_scores, int* __restrict__ token_ids,
                             float* __restrict__ token_alpha, float* __restrict__ shared_scale) {
    static_assert(Tokens >= kSparseMoeSmallTMin && Tokens <= kSparseMoeSmallTMax);
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

    const int token = tid >> 5;
    if (token < Tokens) {
        sparse_moe_select_top8_warp(scores[token], token_ids + token * kTopK,
                                    token_alpha + token * kTopK, shared_scale + token,
                                    selected_logits[token]);
    }
}

template <int Tokens>
void launch_s1(const Tensor& x, const Weight& router, const SparseMoeSmallTWorkspace& workspace,
               cudaStream_t stream) {
    sparse_moe_small_t_s1_kernel<Tokens>
        <<<kRouterRows * kRouterPartitions, kRouterWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(router.qdata),
            static_cast<float*>(workspace.scratch.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Tokens>
void launch_s2(const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    sparse_moe_small_t_s2_kernel<Tokens>
        <<<1, Tokens * 32, 0, stream>>>(static_cast<const float*>(workspace.scratch.data),
                                        static_cast<int*>(workspace.token_ids.data),
                                        static_cast<float*>(workspace.token_alpha.data),
                                        static_cast<float*>(workspace.shared_scale.data));
    CUDA_CHECK(cudaGetLastError());
}

SparseMoeDecodeWorkspace token_workspace(const SparseMoeSmallTWorkspace& workspace, int token) {
    SparseMoeDecodeWorkspace out;
    out.ids =
        Tensor(static_cast<int*>(workspace.token_ids.data) + token * kTopK, DType::I32, {kTopK});
    out.alpha = Tensor(static_cast<float*>(workspace.token_alpha.data) + token * kTopK, DType::FP32,
                       {kTopK});
    out.shared_scale =
        Tensor(static_cast<float*>(workspace.shared_scale.data) + token, DType::FP32, {1});
    out.scratch = Tensor(static_cast<float*>(workspace.scratch.data) +
                             static_cast<std::int64_t>(token) * kPaths * kIntermediate,
                         DType::FP32, {kPaths, kIntermediate});
    return out;
}

void launch_s3_token_loop(const Tensor& x, const SparseMoeWeights& weights,
                          const SparseMoeSmallTPlan& plan,
                          const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    const SparseMoeDecodePlan decode_plan =
        resolve_sparse_moe_decode_plan(weights.routed_gate_up.qtype, weights.routed_down.qtype);
    for (int token = 0; token < plan.tokens; ++token) {
        const Tensor x_column                = x.slice(1, token, 1);
        const SparseMoeDecodeWorkspace views = token_workspace(workspace, token);
        sparse_moe_decode_launch_d3(x_column, weights, views, decode_plan.d3, stream);
    }
}

void launch_s4_token_loop(const SparseMoeWeights& weights, Tensor& destination,
                          const SparseMoeSmallTPlan& plan,
                          const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    const SparseMoeDecodePlan decode_plan =
        resolve_sparse_moe_decode_plan(weights.routed_gate_up.qtype, weights.routed_down.qtype);
    for (int token = 0; token < plan.tokens; ++token) {
        Tensor destination_column            = destination.slice(1, token, 1);
        const SparseMoeDecodeWorkspace views = token_workspace(workspace, token);
        sparse_moe_decode_launch_d4(weights, destination_column, views, decode_plan.d4, stream);
    }
}

template <class Launch>
void dispatch_tokens(std::int32_t tokens, Launch&& launch) {
    switch (tokens) {
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
    default:
        throw std::invalid_argument("sparse_moe small-T: unsupported token count");
    }
}

} // namespace

void sparse_moe_small_t_launch_s1(const Tensor& x, const Weight& router_shared_gate,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    dispatch_tokens(x.ne[1], [&]<int Tokens>() {
        launch_s1<Tokens>(x, router_shared_gate, workspace, stream);
    });
}

void sparse_moe_small_t_launch_s2(const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    dispatch_tokens(plan.tokens, [&]<int Tokens>() { launch_s2<Tokens>(workspace, stream); });
}

void sparse_moe_small_t_launch_s3(const Tensor& x, const SparseMoeWeights& weights,
                                  const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    launch_s3_token_loop(x, weights, plan, workspace, stream);
}

void sparse_moe_small_t_launch_s4(const SparseMoeWeights& weights, Tensor& destination,
                                  const SparseMoeSmallTPlan& plan,
                                  const SparseMoeSmallTWorkspace& workspace, cudaStream_t stream) {
    launch_s4_token_loop(weights, destination, plan, workspace, stream);
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
