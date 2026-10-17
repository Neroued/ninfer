#include "ops/moe/kernels/cuda/block_scaled/output.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/moe/kernels/cuda/routing/topk.cuh"

namespace ninfer::ops::detail::moe_cuda::block_scaled {
namespace {
using namespace common;

template <bool Gate, class Output>
__global__ void shared_vector(const __nv_bfloat16* x, const __nv_bfloat16* weights, Storage storage,
                              const float* coefficients, const float* beta, Output output,
                              const __nv_bfloat16* control = nullptr, float* scores = nullptr) {
    const int route_blocks = Gate && control ? (kExperts + 4) / 4 : 0;
    const bool route       = blockIdx.x < route_blocks;
    const int row   = (route ? blockIdx.x : blockIdx.x - route_blocks) * 4 + (threadIdx.x >> 5);
    const int token = blockIdx.y, lane = threadIdx.x & 31;
    if (control && storage.groups.counts && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int e = threadIdx.x; e < kExperts; e += blockDim.x) { storage.groups.counts[e] = 0; }
    }
    if (route && row >= kExperts + 1) { return; }
    float sum = 0, up = 0;
    auto* h = static_cast<float*>(storage.shared_activation);
    if constexpr (Gate) {
        const auto* source = route ? control : weights;
        for (int k = lane * 8; k < kHidden; k += 32 * 8) {
            Bf16x8Bits xv, gv, uv;
            xv.raw = load_vec<uint4>(x + std::int64_t(token) * kHidden + k);
            gv.raw = load_vec<uint4>(source + std::int64_t(row) * kHidden + k);
            if (!route) {
                uv.raw = load_vec<uint4>(weights + std::int64_t(row + kIntermediate) * kHidden + k);
            }
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const auto value = __bfloat1622float2(xv.pair[pair]);
                const auto gate  = __bfloat1622float2(gv.pair[pair]);
                sum              = fmaf(value.x, gate.x, sum);
                sum              = fmaf(value.y, gate.y, sum);
                if (!route) {
                    const auto u = __bfloat1622float2(uv.pair[pair]);
                    up           = fmaf(value.x, u.x, up);
                    up           = fmaf(value.y, u.y, up);
                }
            }
        }
        sum = warp_reduce_sum(sum);
        up  = warp_reduce_sum(up);
        if (!lane) {
            if (route) {
                scores[token * kScoreRows + row] = sum;
            } else {
                h[token * kIntermediate + row] = silu(sum) * up;
            }
        }
    } else {
        for (int k = lane; k < kIntermediate; k += 32) {
            sum = fmaf(h[token * kIntermediate + k],
                       __bfloat162float(weights[std::int64_t(row) * kIntermediate + k]), sum);
        }
        sum = warp_reduce_sum(sum);
        if (!lane) {
            const int i = token * kHidden + row;
            output.store(i, output.initial(i) + routed_value(token, row, storage, coefficients) +
                                beta[token] * sum);
        }
    }
}

template <bool Gate, int Columns, class Output>
__global__ __launch_bounds__(128) void shared_mma(const __nv_bfloat16* x,
                                                  const __nv_bfloat16* weights, Storage storage,
                                                  const float* coefficients, const float* beta,
                                                  Output output, int tokens,
                                                  const __nv_bfloat16* control = nullptr,
                                                  float* scores                = nullptr) {
    constexpr int BK = 64, BN = Columns, Rows = Gate ? 32 : 64, K = Gate ? kHidden : kIntermediate;
    constexpr int WarpColumns = BN / 2, NT = WarpColumns / 8;
    constexpr int Planes = Gate ? 1 : 2;
    __shared__ __align__(16) __nv_bfloat16 ws[2][64 * BK];
    __shared__ __align__(16) __nv_bfloat16 xs[2][Planes][BN * BK];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % 2, wn = warp / 2;
    const int route_blocks = Gate && control ? (kExperts + 64) / 64 : 0;
    const bool route       = blockIdx.x < route_blocks;
    const int row0         = route ? blockIdx.x * 64 : (blockIdx.x - route_blocks) * Rows;
    const int token0       = blockIdx.y * BN;
    if (control && storage.groups.counts && blockIdx.x == 0 && blockIdx.y == 0) {
        for (int e = threadIdx.x; e < kExperts; e += blockDim.x) { storage.groups.counts[e] = 0; }
    }
    auto* h    = static_cast<__nv_bfloat16*>(storage.shared_activation);
    auto stage = [&](int s, int kt) {
        for (int task = threadIdx.x; task < 64 * (BK / 8); task += 128) {
            const int row = task / (BK / 8), k = task % (BK / 8) * 8;
            const int wr =
                row0 + (Gate && !route ? (row < 32 ? row : row - 32 + kIntermediate) : row);
            const bool valid = !route || wr < kExperts + 1;
            const auto* source =
                (route ? control : weights) + std::int64_t(valid ? wr : 0) * K + kt * BK + k;
            cp_async_zfill<16>(ws[s] + row * BK + swizzle_xor_64(row, k), source, valid ? 16 : 0);
        }
        for (int task = threadIdx.x; task < BN * (BK / 8); task += 128) {
            const int row = task / (BK / 8), k = task % (BK / 8) * 8;
            const int token = token0 + row;
#pragma unroll
            for (int p = 0; p < Planes; ++p) {
                const auto* source = Gate ? x : h + std::int64_t(p) * tokens * kIntermediate;
                source += std::int64_t(token < tokens ? token : 0) * K + kt * BK + k;
                cp_async_zfill<16>(xs[s][p] + row * BK + swizzle_xor_64(row, k), source,
                                   token < tokens ? 16 : 0);
            }
        }
        cp_commit();
    };
    stage(0, 0);
    stage(1, 1);
    float acc[2][NT][4] = {};
    for (int kt = 0; kt < K / BK; ++kt) {
        const int s = kt % 2;
        cp_wait<1>();
        __syncthreads();
#pragma unroll
        for (int ki = 0; ki < BK / 16; ++ki) {
            unsigned a[2][4];
#pragma unroll
            for (int m = 0; m < 2; ++m) {
                const int row = (Gate ? wm * 16 + m * 32 : wm * 32 + m * 16) + (lane & 7) +
                                ((lane >> 3 & 1) * 8);
                const int k = ki * 16 + (lane >> 4) * 8;
                ldmatrix_x4(a[m][0], a[m][1], a[m][2], a[m][3],
                            smem_addr(ws[s] + row * BK + swizzle_xor_64(row, k)));
            }
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                const int row = wn * WarpColumns + n * 8 + (lane & 7);
                const int k   = ki * 16 + ((lane >> 3) & 1) * 8;
#pragma unroll
                for (int p = 0; p < Planes; ++p) {
                    unsigned b[2];
                    ldmatrix_x2(b[0], b[1],
                                smem_addr(xs[s][p] + row * BK + swizzle_xor_64(row, k)));
#pragma unroll
                    for (int m = 0; m < 2; ++m) {
                        mma_bf16(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3], a[m][0],
                                 a[m][1], a[m][2], a[m][3], b[0], b[1]);
                    }
                }
            }
        }
        __syncthreads();
        if (kt + 2 < K / BK) {
            stage(s, kt + 2);
        } else {
            cp_commit();
        }
    }
    cp_wait<0>();
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int token = token0 + wn * WarpColumns + n * 8 + (lane & 3) * 2 + (v & 1);
            if (token < tokens) {
                if constexpr (Gate) {
                    if (route) {
#pragma unroll
                        for (int m = 0; m < 2; ++m) {
                            const int row = row0 + wm * 16 + m * 32 + (lane >> 2) + (v / 2) * 8;
                            if (row < kExperts + 1) {
                                scores[token * kScoreRows + row] = acc[m][n][v];
                            }
                        }
                        continue;
                    }
                    const int row     = row0 + wm * 16 + (lane >> 2) + (v / 2) * 8;
                    const float value = silu(acc[0][n][v]) * acc[1][n][v];
                    const auto hi     = __float2bfloat16_rn(value);
                    const int index   = token * kIntermediate + row;
                    h[index]          = hi;
                    h[tokens * kIntermediate + index] =
                        __float2bfloat16_rn(value - __bfloat162float(hi));
                } else {
#pragma unroll
                    for (int m = 0; m < 2; ++m) {
                        const int row = row0 + wm * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                        const int i   = token * kHidden + row;
                        output.store(i, output.initial(i) +
                                            routed_value(token, row, storage, coefficients) +
                                            beta[token] * acc[m][n][v]);
                    }
                }
            }
        }
    }
}

template <bool Count, bool FixedIndices = false>
__global__ void route_selection(const float* scores, int* ids, float* alpha, float* beta,
                                int* counts, int tokens) {
    constexpr int Warps = Count ? 8 : 1;
    __shared__ float logits[Warps][kTopK];
    __shared__ int local[Count ? kExperts : 1];
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int token = blockIdx.x * Warps + warp;
    if constexpr (Count) {
        for (int e = threadIdx.x; e < kExperts; e += blockDim.x) { local[e] = 0; }
        __syncthreads();
    }
    if (token < tokens) {
        routing::select_experts<SwiGLUH2560I640E512K10, FixedIndices>(
            scores + token * kScoreRows, ids + token * kTopK, alpha + token * kTopK, beta + token,
            logits[warp]);
        if constexpr (Count) {
            __syncwarp();
            if (lane < kTopK) { atomicAdd(local + ids[token * kTopK + lane], 1); }
        }
    }
    if constexpr (Count) {
        __syncthreads();
        for (int e = threadIdx.x; e < kExperts; e += blockDim.x) {
            if (local[e]) { atomicAdd(counts + e, local[e]); }
        }
    }
}
} // namespace

void launch_shared(const Tensor& x, const ScaledSwiGLUTerm& shared, const ExpertRoutes& routes,
                   MoeEpilogue mode, Tensor& destination, const Storage& storage, const Plan& plan,
                   CudaExecutionView execution, bool prepared) {
    const auto stream = execution.stream();
    const auto* input = static_cast<const __nv_bfloat16*>(x.data);
    const auto* gu    = static_cast<const __nv_bfloat16*>(shared.weights.gate_up.weight.qdata);
    const auto* down  = static_cast<const __nv_bfloat16*>(shared.weights.down.weight.qdata);
    const auto* coefficients = static_cast<const float*>(routes.coefficients.data);
    const auto* beta         = static_cast<const float*>(shared.coefficients.data);
    const int tokens         = x.ne[1];
    with_output(destination, mode, [&](auto output) {
        // Complete already produced the shared activation alongside control.
        // The final shared down still combines routed and shared terms once.
        if (plan.shared_algorithm == SharedAlgorithm::Vector) {
            if (!prepared) {
                shared_vector<true><<<dim3(kIntermediate / 4, tokens), 128, 0, stream>>>(
                    input, gu, storage, coefficients, beta, output);
            }
            CUDA_CHECK(cudaGetLastError());
            shared_vector<false><<<dim3(kHidden / 4, tokens), 128, 0, stream>>>(
                input, down, storage, coefficients, beta, output);
        } else {
            const auto launch = [&]<int Columns>() {
                if (!prepared) {
                    shared_mma<true, Columns>
                        <<<dim3(kIntermediate / 32, (tokens + Columns - 1) / Columns), 128, 0,
                           stream>>>(input, gu, storage, coefficients, beta, output, tokens);
                }
                CUDA_CHECK(cudaGetLastError());
                shared_mma<false, Columns>
                    <<<dim3(kHidden / 64, (tokens + Columns - 1) / Columns), 128, 0, stream>>>(
                        input, down, storage, coefficients, beta, output, tokens);
            };
            if (plan.shared_columns == 64) {
                launch.template operator()<64>();
            } else {
                launch.template operator()<32>();
            }
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

void launch_control_shared(const Tensor& x, const MoeWeights& weights, const ExpertRoutes& routes,
                           const Tensor& shared, float* scores, const Storage& storage,
                           const Plan& plan, CudaExecutionView execution) {
    const auto stream   = execution.stream();
    const auto* input   = static_cast<const __nv_bfloat16*>(x.data);
    const auto* gu      = static_cast<const __nv_bfloat16*>(weights.shared.gate_up.weight.qdata);
    const auto* control = static_cast<const __nv_bfloat16*>(weights.control.qdata);
    if (plan.shared_algorithm == SharedAlgorithm::Vector) {
        shared_vector<true>
            <<<dim3(kIntermediate / 4 + (kExperts + 4) / 4, x.ne[1]), 128, 0, stream>>>(
                input, gu, storage, nullptr, nullptr, FloatContribution{}, control, scores);
    } else {
        const auto launch = [&]<int Columns>() {
            shared_mma<true, Columns><<<dim3(kIntermediate / 32 + (kExperts + 64) / 64,
                                             (x.ne[1] + Columns - 1) / Columns),
                                        128, 0, stream>>>(input, gu, storage, nullptr, nullptr,
                                                          FloatContribution{}, x.ne[1], control,
                                                          scores);
        };
        if (plan.shared_columns == 64) {
            launch.template operator()<64>();
        } else {
            launch.template operator()<32>();
        }
    }
    CUDA_CHECK(cudaGetLastError());
    const auto select = [&]<bool FixedIndices>() {
        if (storage.groups.counts) {
            route_selection<true, FixedIndices><<<(x.ne[1] + 7) / 8, 256, 0, stream>>>(
                scores, static_cast<int*>(routes.ids.data),
                static_cast<float*>(routes.coefficients.data), static_cast<float*>(shared.data),
                storage.groups.counts, x.ne[1]);
        } else {
            route_selection<false, FixedIndices>
                <<<x.ne[1], 32, 0, stream>>>(scores, static_cast<int*>(routes.ids.data),
                                             static_cast<float*>(routes.coefficients.data),
                                             static_cast<float*>(shared.data), nullptr, x.ne[1]);
        }
    };
    if (plan.network_topk)
        select.template operator()<true>();
    else
        select.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled
