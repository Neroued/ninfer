#include "ops/common/cuda/reduce.cuh"
#include "ops/linear/kernels/bf16/bf16_template_launch.cuh"
#include "ops/moe/topk_sigmoid_route/dispatch.h"

#include <cooperative_groups.h>
#include <algorithm>
#include <climits>

namespace ninfer::ops::detail {
namespace {
namespace cg = cooperative_groups;

constexpr int H = 2304, E = 128, K = 8;

struct RouteOutput {
    const float* bias;
    int* ids;
    float* coefficients;
    int tokens;
    float scale, epsilon;
    moe_cuda::TokenValidity validity;
};

template <class LoadLogit>
__device__ __forceinline__ void finish_route(RouteOutput out, int token, LoadLogit load_logit) {
    constexpr unsigned warp_mask = 0xffffffff;
    const int lane               = threadIdx.x & 31;
    const auto offset            = std::int64_t(token) * K;
    if (!out.validity.contains(token)) {
        if (lane < K) {
            out.ids[offset + lane]          = -1;
            out.coefficients[offset + lane] = 0.0f;
        }
        return;
    }

    unsigned keys[E / 32];
    float probabilities[E / 32];
#pragma unroll
    for (int j = 0; j < E / 32; ++j) {
        const int expert    = j * 32 + lane;
        const float logit   = load_logit(expert);
        const float exp_abs = expf(-fabsf(logit));
        probabilities[j]    = (logit < 0.0f ? exp_abs : 1.0f) / (1.0f + exp_abs);
        keys[j]             = common::ordered_score_bits(probabilities[j] + out.bias[expert]);
    }

    float numerator = 0.0f, denominator = out.epsilon;
    int chosen = -1;
#pragma unroll
    for (int k = 0; k < K; ++k) {
        unsigned local_key      = keys[0];
        int slot                = 0;
        float local_probability = probabilities[0];
#pragma unroll
        for (int j = 1; j < E / 32; ++j) {
            const bool better = keys[j] > local_key;
            local_key         = max(local_key, keys[j]);
            if (better) {
                slot              = j;
                local_probability = probabilities[j];
            }
        }
        const unsigned key = __reduce_max_sync(warp_mask, local_key);
        const int selected =
            __reduce_min_sync(warp_mask, local_key == key ? slot * 32 + lane : INT_MAX);
        const float probability = __shfl_sync(warp_mask, local_probability, selected & 31);
        denominator += probability;
        if (lane == k) {
            chosen    = selected;
            numerator = probability;
        }
#pragma unroll
        for (int j = 0; j < E / 32; ++j)
            if (selected == j * 32 + lane) keys[j] = 0;
    }
    if (lane < K) {
        out.ids[offset + lane]          = chosen;
        out.coefficients[offset + lane] = (numerator / denominator) * out.scale;
    }
}

// CTAs in the cluster publish expert tiles and, for split-K, disjoint K
// contributions. The routing warp sums the partial logits before the sigmoid.
template <int ExpertBlocks, int TileTokens, int Splits = 1>
__device__ __forceinline__ void finish_cluster(RouteOutput out, const float* logits, int first) {
    constexpr int Rows          = E / ExpertBlocks;
    constexpr int ClusterBlocks = ExpertBlocks * Splits;
    auto cluster                = cg::this_cluster();
    cluster.sync();
    const int warp   = threadIdx.x / 32;
    const int stride = ClusterBlocks * (blockDim.x / 32);
    for (int col = cluster.block_rank() + warp * ClusterBlocks; col < TileTokens; col += stride) {
        if (col < out.tokens - first)
            finish_route(out, first + col, [&](int expert) {
                const int expert_block = expert / Rows;
                const int index        = col * Rows + expert % Rows;
                float sum              = cluster.map_shared_rank(logits, expert_block)[index];
#pragma unroll
                for (int split = 1; split < Splits; ++split)
                    sum +=
                        cluster.map_shared_rank(logits, split * ExpertBlocks + expert_block)[index];
                return sum;
            });
    }
    // All peer shared-memory reads must finish before any CTA in the cluster exits.
    cluster.sync();
}

template <int Blocks>
struct SlicedRouting {
    [[maybe_unused]] static constexpr bool kSlicedTile = true;

    template <class Schedule>
    __device__ void finish_sliced_tile(RouteOutput out, float* logits, int, int first, int,
                                       int) const {
        finish_cluster<Blocks, Schedule::kBlockTokens>(out, logits, first);
    }
};

template <class Schedule, int Blocks>
__global__ __launch_bounds__(Schedule::kThreads) void __cluster_dims__(Blocks, 1, 1)
    fused_sliced(const __nv_bfloat16* x, const __nv_bfloat16* weight, RouteOutput out) {
    extern __shared__ __align__(16) unsigned char shared[];
    const int first = (blockIdx.x / Blocks) * Schedule::kBlockTokens;
    bf16_a16_sliced_k_mma_body<Schedule>(x, weight, out, SlicedRouting<Blocks>{}, E, H, out.tokens,
                                         blockIdx.x % Blocks, first, shared);
}

template <int Blocks, int Splits>
struct TmaRouting {
    [[maybe_unused]] static constexpr bool kTmaCollective = true;

    template <class Schedule, bool>
    __device__ void finish_tile(RouteOutput out, unsigned char* storage,
                                float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int,
                                int first, int, int) const {
        constexpr int Rows = Schedule::kBlockRows;
        auto* logits       = reinterpret_cast<float*>(storage);
        const int tid      = int(threadIdx.x) - Schedule::kProducerThreads;
        if (tid >= 0) {
            const int warp = tid / 32;
            const int row  = warp / Schedule::kWarpsTokens * Schedule::kWarpRows;
            const int col  = warp % Schedule::kWarpsTokens * Schedule::kWarpTokens;
#pragma unroll
            for (int m = 0; m < Schedule::kMmaRows; ++m) {
#pragma unroll
                for (int n = 0; n < Schedule::kMmaTokens; ++n) {
                    const auto& c = accum[m][n];
                    const auto q  = common::mma_row_quad(make_float4(c[0], c[1], c[2], c[3]));
                    common::store_vec(
                        logits + (col + n * 8 + q.column) * Rows + row + m * 16 + q.row, q.values);
                }
            }
        }
        finish_cluster<Blocks, Schedule::kBlockTokens, Splits>(out, logits, first);
    }
};

template <class Schedule, int Blocks, int Splits>
__global__ __launch_bounds__(Schedule::kThreads) void __cluster_dims__(Blocks, 1, Splits)
    fused_tma(const __grid_constant__ Bf16TmaDescriptors maps, RouteOutput out, int offset,
              int count) {
    using Epilogue = TmaRouting<Blocks, Splits>;
    bf16_a16_tma_mma_body<Schedule, false, RouteOutput, Epilogue, Splits>(maps, out, Epilogue{}, E,
                                                                          H, offset, count);
}

struct LogitsOutput {
    float* data;
    int tokens;
};

template <int Splits>
struct LogitsStore {
    __device__ void store_fragment(LogitsOutput output, int row, int token, float4 values, int,
                                   int tokens) const {
        const int lane = threadIdx.x & 31;
        const auto q   = common::mma_row_quad(values);
        const int r    = row - (lane >> 2) + q.row;
        const int t    = token - 2 * (lane & 3) + q.column;
        if (t < tokens) {
            auto* plane = output.data;
            if constexpr (Splits > 1) plane += std::int64_t(blockIdx.z) * output.tokens * E;
            common::store_vec(plane + std::int64_t(t) * E + r, q.values);
        }
    }
};

template <int Warps, int Splits>
__global__ __launch_bounds__(Warps * 32) void route_logits(const float* logits, RouteOutput out) {
    const int token = blockIdx.x * Warps + threadIdx.x / 32;
    if (token < out.tokens) {
        const auto* column = logits + std::int64_t(token) * E;
        finish_route(out, token, [&](int expert) {
            float sum = column[expert];
#pragma unroll
            for (int split = 1; split < Splits; ++split)
                sum += column[std::int64_t(split) * out.tokens * E + expert];
            return sum;
        });
    }
}

template <class Schedule>
using Static   = Bf16ScheduleInstance<Schedule, H>;
using Sliced8  = Static<Bf16A16SlicedKMmaSchedule<16, 8, 8, 96, 2>>;
using Sliced16 = Static<Bf16A16SlicedKMmaSchedule<32, 16, 8, 48, 2>>;

template <int Rows, int Tokens, int TileK, int WarpRows, int WarpTokens, int Stages>
using Tma = Static<Bf16A16TmaMmaSchedule<Rows, Tokens, TileK, WarpRows, WarpTokens, Stages, 1,
                                         Bf16MmaRaster::RowFast>>;
using FusedSplit32  = Tma<32, 32, 64, 16, 16, 3>;
using SplitLogits48 = Tma<64, 48, 64, 32, 24, 3>;
using SplitLogits64 = Tma<64, 64, 64, 32, 32, 2>;
using Logits64      = Tma<32, 64, 128, 16, 16, 3>;
using Fused48       = Tma<64, 48, 64, 16, 8, 5>;
using Fused56       = Tma<64, 56, 64, 16, 8, 3>;
using WideLogits64  = Tma<64, 64, 128, 16, 32, 3>;

template <class Schedule>
void launch_sliced(const Bf16A16Operands& p, float*, RouteOutput out, cudaStream_t stream) {
    constexpr int Blocks     = E / Schedule::kBlockRows;
    constexpr int TileTokens = Schedule::kBlockTokens;
    constexpr auto kernel    = fused_sliced<Schedule, Blocks>;
    const int shared =
        common::prepare_shared<linear_mma_shared_bytes<Schedule, SlicedRouting<Blocks>>, kernel,
                               true>();
    kernel<<<common::div_up(p.tokens, TileTokens) * Blocks, Schedule::kThreads, shared, stream>>>(
        p.x, p.weight, out);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, int Splits = 1>
void launch_fused_tma(const Bf16A16Operands& p, float*, RouteOutput out, cudaStream_t stream) {
    constexpr int Blocks     = E / Schedule::kBlockRows;
    constexpr int TileTokens = Schedule::kBlockTokens;
    constexpr auto kernel    = fused_tma<Schedule, Blocks, Splits>;
    static_assert(Schedule::kRaster == Bf16MmaRaster::RowFast);
    static_assert(H % (Splits * Schedule::kBlockK) == 0);
    constexpr int bytes =
        bf16_tma_scratch_bytes<Schedule, TmaRouting<Blocks, Splits>> + Schedule::kBarrierBytes;
    const auto maps = make_bf16_tma_descriptors<Schedule>(p);
    common::prepare_shared<bytes, kernel, true>();
    common::for_each_token_slice(p.tokens, TileTokens, [&](int offset, int count) {
        const dim3 grid(common::div_up(count, TileTokens) * Blocks, 1, Splits);
        kernel<<<grid, Schedule::kThreads, bytes, stream>>>(maps, out, offset, count);
        CUDA_CHECK(cudaGetLastError());
    });
}

template <class Schedule, int Warps, int Splits = 1>
void launch_staged(const Bf16A16Operands& p, float* logits, RouteOutput out, cudaStream_t stream) {
    launch_bf16_a16_tma_mma<Schedule, Splits>(p, LogitsOutput{logits, p.tokens},
                                              LogitsStore<Splits>{}, stream);
    route_logits<Warps, Splits>
        <<<common::div_up(p.tokens, Warps), Warps * 32, 0, stream>>>(logits, out);
    CUDA_CHECK(cudaGetLastError());
}

struct Plan {
    int last_token_count;
    int logits_planes;
    void (*launch)(const Bf16A16Operands&, float*, RouteOutput, cudaStream_t);
};

template <class Schedule, int Warps, int Splits = 1>
constexpr Plan staged_plan(int last_token_count) {
    return {last_token_count, Splits, launch_staged<Schedule, Warps, Splits>};
}

constexpr Plan kPlans[]{
    {144, 0, launch_sliced<Sliced8>},
    {416, 0, launch_sliced<Sliced16>},
    {832, 0, launch_fused_tma<FusedSplit32, 2>},
    staged_plan<SplitLogits48, 4, 3>(1344),
    staged_plan<SplitLogits64, 4, 3>(1792),
    staged_plan<Logits64, 8>(2688),
    {4032, 0, launch_fused_tma<Fused48>},
    {4256, 0, launch_fused_tma<Fused56>},
    staged_plan<WideLogits64, 4>(INT_MAX),
};
} // namespace

const RouteProfile moe_sigmoid::registered_route{H, E, K};

std::size_t moe_sigmoid::workspace_capacity_bytes(int first, int last) {
    std::size_t peak = 0;
    for (const auto& plan : kPlans) {
        if (first <= plan.last_token_count) {
            const int tokens = std::min(last, plan.last_token_count);
            peak = std::max(peak, std::size_t(tokens) * E * plan.logits_planes * sizeof(float));
        }
        if (last <= plan.last_token_count) break;
    }
    return peak;
}

void moe_sigmoid::launch(const Tensor& x, const Weight& w, const Tensor& bias, ExpertRoutes& routes,
                         const SigmoidRouteProfile& profile, float* logits,
                         moe_cuda::TokenValidity validity, cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, w);
    const RouteOutput out{static_cast<const float*>(bias.data),
                          static_cast<int*>(routes.ids.data),
                          static_cast<float*>(routes.coefficients.data),
                          p.tokens,
                          profile.scaling_factor,
                          profile.normalization_epsilon,
                          validity};
    const auto plan = std::ranges::lower_bound(kPlans, p.tokens, {}, &Plan::last_token_count);
    plan->launch(p, logits, out, stream);
}
} // namespace ninfer::ops::detail
