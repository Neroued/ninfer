#include "ops/moe/kernels/cuda/routing/launch.h"
#include "core/pdl.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/moe/kernels/cuda/routing/router_mma.cuh"
#include "ops/moe/kernels/cuda/routing/topk.cuh"

#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::routing {
namespace {
using namespace common;

__device__ __forceinline__ float dot_eight(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    const uint4 av = load_vec<uint4>(a), bv = load_vec<uint4>(b);
    const float2 a0 = bf16x2_bits_to_float2(av.x), a1 = bf16x2_bits_to_float2(av.y);
    const float2 a2 = bf16x2_bits_to_float2(av.z), a3 = bf16x2_bits_to_float2(av.w);
    const float2 b0 = bf16x2_bits_to_float2(bv.x), b1 = bf16x2_bits_to_float2(bv.y);
    const float2 b2 = bf16x2_bits_to_float2(bv.z), b3 = bf16x2_bits_to_float2(bv.w);
    float sum = fmaf(a0.x, b0.x, 0.0F);
    sum       = fmaf(a0.y, b0.y, sum);
    sum       = fmaf(a1.x, b1.x, sum);
    sum       = fmaf(a1.y, b1.y, sum);
    sum       = fmaf(a2.x, b2.x, sum);
    sum       = fmaf(a2.y, b2.y, sum);
    sum       = fmaf(a3.x, b3.x, sum);
    return fmaf(a3.y, b3.y, sum);
}

__global__ void shared_gate(const __nv_bfloat16* x, const __nv_bfloat16* weight, float* beta,
                            int hidden, int tokens) {
    const int token = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane  = threadIdx.x % 32;
    if (token >= tokens) { return; }
    float sum = 0;
    for (int k = lane * 8; k < hidden; k += 32 * 8) {
        sum += dot_eight(x + std::int64_t(token) * hidden + k, weight + k);
    }
    sum = warp_reduce_sum(sum);
    if (!lane) { beta[token] = 1.0F / (1.0F + __expf(-sum)); }
}

template <class Shape>
__global__ void router_vector(const __nv_bfloat16* x, const __nv_bfloat16* weights, float* scores,
                              MoeExecutionHints hints) {
    __shared__ float partial[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const auto* row = weights + std::int64_t(blockIdx.x) * Shape::hidden;
    float sum       = 0;
#pragma unroll
    for (int k = threadIdx.x * 8; k < Shape::hidden; k += 256 * 8) {
        sum += dot_eight(row + k, x + k);
    }
    sum = warp_reduce_sum(sum);
    if (!lane) { partial[warp] = sum; }
    __syncthreads();
    if (!warp) {
        const float value = warp_reduce_sum<8>(lane < 8 ? partial[lane] : 0);
        if (!lane) { scores[blockIdx.x] = value; }
    }
    common::prefetch_l2(hints.read_ahead, hints.read_ahead_bytes,
                        (std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128);
}

template <class Shape, int Tokens>
__global__ void router_token_tile(const __nv_bfloat16* x, const __nv_bfloat16* weights,
                                  float* scores, bool programmatic) {
    constexpr int Partitions = 4, Warps = 4;
    __shared__ float partial[Warps][Tokens];
    if (!threadIdx.x && programmatic) { pdl::trigger_dependents(); }
    const int row = blockIdx.x / Partitions, partition = blockIdx.x % Partitions;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int k       = partition * (Shape::hidden / Partitions) + threadIdx.x * 4;
    const auto weight = load_vec<uint2>(weights + std::int64_t(row) * Shape::hidden + k);
    const auto w0 = bf16x2_bits_to_float2(weight.x), w1 = bf16x2_bits_to_float2(weight.y);
    float accumulator[Tokens];
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const auto input = load_vec<uint2>(x + std::int64_t(token) * Shape::hidden + k);
        const auto x0 = bf16x2_bits_to_float2(input.x), x1 = bf16x2_bits_to_float2(input.y);
        float sum          = fmaf(w0.x, x0.x, 0.0F);
        sum                = fmaf(w0.y, x0.y, sum);
        sum                = fmaf(w1.x, x1.x, sum);
        accumulator[token] = fmaf(w1.y, x1.y, sum);
    }
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
        const float sum = warp_reduce_sum(accumulator[token]);
        if (!lane) { partial[warp][token] = sum; }
    }
    __syncthreads();
    if (!warp) {
        for (int token = lane; token < Tokens; token += 32) {
            float sum = 0;
#pragma unroll
            for (int w = 0; w < Warps; ++w) { sum += partial[w][token]; }
            scores[(std::int64_t(token) * Shape::control_rows + row) * Partitions + partition] =
                sum;
        }
    }
}

template <class Shape, bool Partial, bool FixedIndices = false>
__global__ void select_routes(const float* scores, int* ids, float* alpha, float* beta,
                              bool programmatic) {
    __shared__ float reduced[Shape::control_rows], selected[Shape::top_k];
    const int token = blockIdx.x, tid = threadIdx.x;
    if (!tid && programmatic) { pdl::trigger_dependents(); }
    if constexpr (Partial) {
        if (programmatic) { pdl::wait_for_dependencies(); }
        for (int row = tid; row < Shape::control_rows; row += blockDim.x) {
            float sum = 0;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                sum += scores[(std::int64_t(token) * Shape::control_rows + row) * 4 + p];
            }
            reduced[row] = sum;
        }
        __syncthreads();
    }
    if (tid < 32) {
        select_experts<Shape, FixedIndices>(
            Partial ? reduced : scores + std::int64_t(token) * Shape::score_rows,
            ids + token * Shape::top_k, alpha + token * Shape::top_k, beta + token, selected);
    }
}

template <class Shape>
__global__ void select_count_routes(const float* scores, int* ids, float* alpha, float* beta,
                                    int* local, int* tile_counts, int tokens) {
    constexpr int TileTokens = 8;
    __shared__ int counts[Shape::experts];
    __shared__ float selected[TileTokens][Shape::top_k];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    for (int e = tid; e < Shape::experts; e += blockDim.x) { counts[e] = 0; }
    __syncthreads();
    const int token = blockIdx.x * TileTokens + warp;
    if (token < tokens) {
        select_experts<Shape>(scores + std::int64_t(token) * Shape::score_rows,
                              ids + token * Shape::top_k, alpha + token * Shape::top_k,
                              beta + token, selected[warp]);
        __syncwarp();
        if (!lane) {
#pragma unroll
            for (int j = 0; j < Shape::top_k; ++j) {
                const int a = token * Shape::top_k + j;
                local[a]    = atomicAdd(counts + ids[a], 1);
            }
        }
    }
    __syncthreads();
    for (int e = tid; e < Shape::experts; e += blockDim.x) {
        tile_counts[std::int64_t(blockIdx.x) * Shape::experts + e] = counts[e];
    }
}

template <int Token = 1, class Function>
void token_extent(int tokens, Function&& launch) {
    if (tokens == Token) {
        launch.template operator()<Token>();
        return;
    }
    if constexpr (Token < 64) {
        token_extent<Token + 1>(tokens, launch);
    } else {
        throw std::logic_error("router token tile exceeds its selected extent");
    }
}
} // namespace

template <class S>
void launch_route_shape(const Tensor& x, const Weight& control, const ExpertRoutes& routes,
                        const Tensor& shared, const RouterPlan& plan, const RouteStorage& storage,
                        const MoeExecutionHints& hints, CudaExecutionView execution) {
    const auto* input   = static_cast<const __nv_bfloat16*>(x.data);
    const auto* weights = static_cast<const __nv_bfloat16*>(control.qdata);
    const auto stream   = execution.stream();
    const int tokens    = x.ne[1];
    if (plan.algorithm == RouterAlgorithm::Vector) {
        router_vector<S>
            <<<S::control_rows, 256, 0, stream>>>(input, weights, storage.scores, hints);
    } else if (plan.algorithm == RouterAlgorithm::TokenTile) {
        if constexpr (S::hidden == 2048) {
            token_extent(tokens, [&]<int T>() {
                router_token_tile<S, T><<<S::control_rows * 4, 128, 0, stream>>>(
                    input, weights, storage.scores, plan.programmatic);
            });
        } else {
            throw std::logic_error("router token tile is unsupported for this shape");
        }
    } else {
        if (plan.mma_rows == 64) {
            if constexpr (S::hidden == 2560) {
                router_mma_kernel<S, 64>
                    <<<dim3((S::control_rows + 63) / 64, (tokens + 63) / 64), 128, 0, stream>>>(
                        input, weights, storage.scores, tokens);
            } else {
                throw std::logic_error("router64 is unsupported for this geometry");
            }
        } else if (plan.mma_rows == 16) {
            router_mma_kernel<S>
                <<<dim3((S::control_rows + 15) / 16, (tokens + 63) / 64), 256, 0, stream>>>(
                    input, weights, storage.scores, tokens);
        } else {
            throw std::logic_error("invalid router row tile");
        }
    }
    CUDA_CHECK(cudaGetLastError());
    auto* ids   = static_cast<int*>(routes.ids.data);
    auto* alpha = static_cast<float*>(routes.coefficients.data);
    auto* beta  = static_cast<float*>(shared.data);
    if (plan.count_assignments) {
        select_count_routes<S><<<(tokens + 7) / 8, 256, 0, stream>>>(
            storage.scores, ids, alpha, beta, storage.local_ranks, storage.tile_counts, tokens);
        CUDA_CHECK(cudaGetLastError());
    } else if (plan.algorithm == RouterAlgorithm::TokenTile) {
        if (plan.programmatic) {
            CUDA_CHECK(pdl::launch_dependent({dim3(tokens), dim3(128), 0, stream},
                                             select_routes<S, true>, storage.scores, ids, alpha,
                                             beta, true));
        } else {
            select_routes<S, true>
                <<<tokens, 128, 0, stream>>>(storage.scores, ids, alpha, beta, false);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        if (plan.network_topk) {
            select_routes<S, false, true>
                <<<tokens, 32, 0, stream>>>(storage.scores, ids, alpha, beta, plan.programmatic);
        } else {
            select_routes<S, false>
                <<<tokens, 32, 0, stream>>>(storage.scores, ids, alpha, beta, plan.programmatic);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch_route(const Tensor& x, const Weight& control, const ExpertRoutes& routes,
                  const Tensor& shared, const RouterPlan& plan, const RouteStorage& storage,
                  const MoeExecutionHints& hints, CudaExecutionView execution) {
    if (control.k == 2560) {
        launch_route_shape<SwiGLUH2560I640E512K10>(x, control, routes, shared, plan, storage, hints,
                                                   execution);
    } else {
        launch_route_shape<SwiGLUH2048I512E256K8>(x, control, routes, shared, plan, storage, hints,
                                                  execution);
    }
}

void launch_shared_gate(const Tensor& x, const Weight& control, const Tensor& shared,
                        CudaExecutionView execution) {
    const auto* weights = static_cast<const __nv_bfloat16*>(control.qdata);
    shared_gate<<<(x.ne[1] + 3) / 4, 128, 0, execution.stream()>>>(
        static_cast<const __nv_bfloat16*>(x.data), weights + (control.n - 1) * control.k,
        static_cast<float*>(shared.data), control.k, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::moe_cuda::routing
