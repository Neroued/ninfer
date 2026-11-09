#include "ops/sampling/candidate_path/launch.h"
#include "core/device.h"
#include "ops/common/cuda/pipeline.cuh"
#include <math_constants.h>
#include "ops/common/cuda/reduce.cuh"
#include "ops/sampling/kernels/random.cuh"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::candidate_path_detail {
using namespace common;
using sampling_detail::sampling_uniform;

namespace {
constexpr int kCandidates = 16, kRank = 256;

struct DeviceArgs {
    const std::int32_t* ids;
    const float* unary;
    const __nv_bfloat16* hidden;
    const std::int32_t* anchors;
    const __nv_bfloat16* predecessor;
    const __nv_bfloat16* successor;
    const std::int32_t* positions;
    const SamplingConfig* configs;
    std::int32_t* drafts;
    float* q;
    int steps;
    int batch;
    const unsigned long long* keys;
    std::int32_t* merged_ids;
};

struct alignas(16) SelectorShared {
    __nv_bfloat16 successors[kCandidates * kRank];
    float product[kRank];
    float edge[kCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
    int ids[kCandidates];
    float unary[kCandidates];
};

// Each local list is descending. Reverse the second half to form a bitonic
// sequence, then merge in five warp exchanges without decoding or narrowing keys.
__device__ unsigned long long merged_key(const DeviceArgs& a, int column) {
    const int lane = threadIdx.x & 31;
    const int rank = lane / 16, local = rank ? 31 - lane : lane;
    auto key = a.keys[(std::int64_t(rank) * a.steps * a.batch + column) * 16 + local];
#pragma unroll
    for (int distance = 16; distance; distance /= 2) {
        const auto other = __shfl_xor_sync(kFullWarpMask, key, distance);
        key = (lane & distance) == 0 ? (key > other ? key : other) : (key < other ? key : other);
    }
    return key;
}

__device__ void merge_current(const DeviceArgs& a, int column, SelectorShared& shared,
                              bool publish) {
    const int lane = threadIdx.x & 31;
    const auto key = merged_key(a, column);
    if (lane < kCandidates) {
        shared.ids[lane]   = id_from_order_key(key);
        shared.unary[lane] = key ? score_from_order_key(key) : -CUDART_INF_F;
        if (publish) a.merged_ids[column * kCandidates + lane] = shared.ids[lane];
    }
}

// The probabilities written here are the same FP32 values consumed by the draw.
__device__ int draw_rank(float edge, float temperature, unsigned long long seed, int position,
                         float* q) {
    const int lane      = threadIdx.x & 31;
    const float maximum = warp_max(edge);
    if (temperature <= 0.0F) {
        const unsigned winners =
            __ballot_sync(kFullWarpMask, lane < kCandidates && edge == maximum);
        const int selected = winners == 0 ? 0 : __ffs(winners) - 1;
        if (lane < kCandidates) q[lane] = lane == selected ? 1.0F : 0.0F;
        return selected;
    }
    const float weight      = lane < kCandidates ? __expf((edge - maximum) / temperature) : 0.0F;
    const float probability = weight / warp_sum(weight);
    if (lane < kCandidates) q[lane] = probability;
    float uniform =
        lane == 0 ? sampling_uniform(seed, position, kSamplePurposeDFlash2Proposal, 0U) : 0.0F;
    uniform          = __shfl_sync(kFullWarpMask, uniform, 0);
    float cumulative = probability;
#pragma unroll
    for (int offset = 1; offset < kCandidates; offset *= 2) {
        const float previous = __shfl_up_sync(kFullWarpMask, cumulative, offset);
        if (lane >= offset) cumulative += previous;
    }
    const unsigned hits =
        __ballot_sync(kFullWarpMask, lane < kCandidates && probability > 0 && uniform < cumulative);
    const unsigned positive = __ballot_sync(kFullWarpMask, lane < kCandidates && probability > 0);
    return hits ? __ffs(hits) - 1 : 31 - __clz(positive);
}

template <bool Gathered, int Warps, bool WaitForPredecessor = true>
__device__ void score_row(const DeviceArgs& a, int column, SelectorShared& shared, const int* ids,
                          const float* unary) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
#pragma unroll
    for (int c = warp; c < kCandidates; c += Warps) {
        int token = lane == 0 ? ids[c] : 0;
        token     = __shfl_sync(kFullWarpMask, token, 0);
        if constexpr (Gathered)
            cp_async_zfill<16, Cache::cg>(
                &shared.successors[c * kRank + lane * 8],
                a.successor + std::int64_t(token == INT_MAX ? 0 : token) * kRank + lane * 8,
                token == INT_MAX ? 0 : 16);
        else
            cp_async<16, Cache::cg>(&shared.successors[c * kRank + lane * 8],
                                    a.successor + std::int64_t(token) * kRank + lane * 8);
    }
    cp_commit();
    if constexpr (WaitForPredecessor) __syncthreads();
    for (int r = tid; r < kRank; r += Warps * 32)
        shared.product[r] =
            __bfloat162float(a.predecessor[std::int64_t(shared.predecessor) * kRank + r]) *
            __bfloat162float(a.hidden[std::int64_t(column) * kRank + r]);
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int c = warp; c < kCandidates; c += Warps) {
        float sum = 0;
#pragma unroll
        for (int r = lane; r < kRank; r += 32)
            sum = fmaf(shared.product[r], __bfloat162float(shared.successors[c * kRank + r]), sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = unary[c] + sum;
    }
    __syncthreads();
}

template <bool Gathered, int Warps>
__device__ void walk_path(const DeviceArgs& a, SelectorShared& shared, const int* all_ids,
                          const float* all_unary) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, batch = blockIdx.x;
    if (tid == 0) {
        shared.predecessor   = a.anchors[batch];
        shared.base_position = a.positions[batch];
        shared.temperature   = a.configs[batch].temperature;
        shared.seed          = a.configs[batch].seed;
    }
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const int* ids   = Gathered ? all_ids + step * kCandidates : a.ids + column * kCandidates;
        const float* unary =
            Gathered ? all_unary + step * kCandidates : a.unary + column * kCandidates;
        score_row<Gathered, Warps>(a, column, shared, ids, unary);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed,
                                           shared.base_position + step, a.q + column * kCandidates);
            if (lane == 0) {
                shared.predecessor = ids[selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

template <bool Gathered, int Warps = 16>
__global__ __launch_bounds__(Warps * 32,
                             1) void selector_walk_kernel(const __grid_constant__ DeviceArgs a) {
    __shared__ SelectorShared shared;
    if constexpr (Gathered) {
        // All columns are independent until path sampling. Merge them in parallel,
        // allowing successor prefetch for the next step to overlap the prior draw.
        __shared__ int ids[15 * kCandidates];
        __shared__ float unary[15 * kCandidates];
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        for (int step = warp; step < a.steps; step += Warps) {
            const int column = blockIdx.x * a.steps + step;
            const auto key   = merged_key(a, column);
            if (lane < kCandidates) {
                ids[step * kCandidates + lane]   = id_from_order_key(key);
                unary[step * kCandidates + lane] = key ? score_from_order_key(key) : -CUDART_INF_F;
                a.merged_ids[column * kCandidates + lane] = ids[step * kCandidates + lane];
            }
        }
        __syncthreads();
        walk_path<Gathered, Warps>(a, shared, ids, unary);
    } else {
        walk_path<Gathered, Warps>(a, shared, nullptr, nullptr);
    }
}

template <bool Gathered, int Warps = 16>
__global__ __launch_bounds__(Warps * 32,
                             2) void selector_lattice_kernel(const __grid_constant__ DeviceArgs a,
                                                             float* edges) {
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if constexpr (Gathered) {
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (warp == 0) merge_current(a, column, shared, p == 0);
        if (step > 0 && warp == 1) {
            const auto key = __shfl_sync(kFullWarpMask, merged_key(a, column - 1), p);
            if (lane == 0) shared.predecessor = id_from_order_key(key);
        } else if (step == 0 && threadIdx.x == 0)
            shared.predecessor = a.anchors[batch];
        __syncthreads();
        if (shared.predecessor == INT_MAX) return;
    } else if (threadIdx.x == 0)
        shared.predecessor = step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
    score_row<Gathered, Warps, !Gathered>(a, column, shared,
                                          Gathered ? shared.ids : a.ids + column * kCandidates,
                                          Gathered ? shared.unary : a.unary + column * kCandidates);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

__global__
__launch_bounds__(32) void selector_lattice_walk_kernel(const __grid_constant__ DeviceArgs a,
                                                        const float* edges) {
    const int lane = threadIdx.x, batch = blockIdx.x;
    const auto seed         = a.configs[batch].seed;
    const float temperature = a.configs[batch].temperature;
    const int position      = a.positions[batch];
    int predecessor_rank    = 0;
#pragma unroll 1
    for (int step = 0; step < a.steps; ++step) {
        const int column = batch * a.steps + step;
        const float edge =
            lane < kCandidates
                ? edges[(static_cast<std::int64_t>(column) * kCandidates + predecessor_rank) *
                            kCandidates +
                        lane]
                : -CUDART_INF_F;
        int selected =
            draw_rank(edge, temperature, seed, position + step, a.q + column * kCandidates);
        predecessor_rank = selected;
        if (lane == 0) a.drafts[column] = a.ids[column * kCandidates + selected];
    }
}

} // namespace

void launch_candidate_path(SelectorPlan plan, CandidateSelectorInput input,
                           const Tensor& candidate_ids, const Tensor& unary_scores,
                           const Tensor& keys, const Tensor& projected_hidden,
                           const Tensor& anchors, const Tensor& predecessor_codebook,
                           const Tensor& successor_codebook, const Tensor& base_positions,
                           const SamplingConfig* configs, Tensor& drafts, Tensor& proposal_q,
                           const SelectorWorkspace& workspace, cudaStream_t stream) {
    const DeviceArgs args{static_cast<const std::int32_t*>(candidate_ids.data),
                          static_cast<const float*>(unary_scores.data),
                          static_cast<const __nv_bfloat16*>(projected_hidden.data),
                          static_cast<const std::int32_t*>(anchors.data),
                          static_cast<const __nv_bfloat16*>(predecessor_codebook.data),
                          static_cast<const __nv_bfloat16*>(successor_codebook.data),
                          static_cast<const std::int32_t*>(base_positions.data),
                          configs,
                          static_cast<std::int32_t*>(drafts.data),
                          static_cast<float*>(proposal_q.data),
                          candidate_ids.ne[1],
                          candidate_ids.ne[2],
                          static_cast<const unsigned long long*>(keys.data),
                          input == CandidateSelectorInput::GatheredKeys
                              ? static_cast<int*>(candidate_ids.data)
                              : nullptr};
    const auto launch = [&]<bool Gathered, int Warps>() {
        if (plan.route == SelectorRoute::Direct) {
            selector_walk_kernel<Gathered, Warps>
                <<<candidate_ids.ne[2], Warps * 32, 0, stream>>>(args);
        } else {
            auto* edges = static_cast<float*>(workspace.edges.data);
            selector_lattice_kernel<Gathered, Warps>
                <<<dim3(args.steps * candidate_ids.ne[2], kCandidates), Warps * 32, 0, stream>>>(
                    args, edges);
            CUDA_CHECK(cudaGetLastError());
            selector_lattice_walk_kernel<<<candidate_ids.ne[2], 32, 0, stream>>>(args, edges);
        }
    };
    if (input == CandidateSelectorInput::GatheredKeys) {
        if (plan.warps == 8)
            launch.template operator()<true, 8>();
        else
            launch.template operator()<true, 16>();
    } else
        launch.template operator()<false, 16>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::candidate_path_detail
