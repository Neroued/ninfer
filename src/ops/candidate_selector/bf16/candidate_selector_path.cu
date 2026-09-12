#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/sampling_device.cuh"
#include <cuda_bf16.h>
#include <stdexcept>

namespace ninfer::ops::detail {
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
};

struct alignas(16) SelectorShared {
    __nv_bfloat16 successors[kCandidates * kRank];
    float product[kRank];
    float edge[kCandidates];
    int predecessor, base_position;
    float temperature;
    unsigned long long seed;
};

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
    const unsigned hits = __ballot_sync(kFullWarpMask, lane < kCandidates && uniform < cumulative);
    return hits ? __ffs(hits) - 1 : kCandidates - 1;
}

__device__ void score_row(const DeviceArgs& a, int column, SelectorShared& shared) {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int token = lane == 0 ? a.ids[column * kCandidates + warp] : 0;
    token     = __shfl_sync(kFullWarpMask, token, 0);
    cp_async<16, Cache::cg>(&shared.successors[warp * kRank + lane * 8],
                            a.successor + static_cast<std::int64_t>(token) * kRank + lane * 8);
    cp_commit();
    // Publish the preceding draw before reading its token. Successor prefetch is independent.
    __syncthreads();
    if (tid < kRank)
        shared.product[tid] =
            __bfloat162float(
                a.predecessor[static_cast<std::int64_t>(shared.predecessor) * kRank + tid]) *
            __bfloat162float(a.hidden[static_cast<std::int64_t>(column) * kRank + tid]);
    cp_wait<0>();
    __syncthreads();
    {
        const int c = warp;
        float sum   = 0;
#pragma unroll
        for (int r = lane; r < kRank; r += 32)
            sum = fmaf(shared.product[r], __bfloat162float(shared.successors[c * kRank + r]), sum);
        sum = warp_reduce_sum(sum);
        if (lane == 0) shared.edge[c] = a.unary[column * kCandidates + c] + sum;
    }
    __syncthreads();
}

__global__ __launch_bounds__(512, 1) void selector_walk_kernel(DeviceArgs a) {
    __shared__ SelectorShared shared;
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
        score_row(a, column, shared);
        if (warp == 0) {
            const float edge   = lane < kCandidates ? shared.edge[lane] : -CUDART_INF_F;
            const int selected = draw_rank(edge, shared.temperature, shared.seed,
                                           shared.base_position + step, a.q + column * kCandidates);
            if (lane == 0) {
                shared.predecessor = a.ids[column * kCandidates + selected];
                a.drafts[column]   = shared.predecessor;
            }
        }
    }
}

__global__ __launch_bounds__(512, 2) void selector_lattice_kernel(DeviceArgs a, float* edges) {
    const int column = blockIdx.x, p = blockIdx.y;
    const int step = column % a.steps, batch = column / a.steps;
    if (step == 0 && p != 0) return;
    __shared__ SelectorShared shared;
    if (threadIdx.x == 0)
        shared.predecessor = step == 0 ? a.anchors[batch] : a.ids[(column - 1) * kCandidates + p];
    score_row(a, column, shared);
    if (threadIdx.x < kCandidates)
        edges[(static_cast<std::int64_t>(column) * kCandidates + p) * kCandidates + threadIdx.x] =
            shared.edge[threadIdx.x];
}

__global__ __launch_bounds__(32) void selector_lattice_walk_kernel(DeviceArgs a,
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

void candidate_selector_path_launch(SelectorRoute route, const Tensor& candidate_ids,
                                    const Tensor& unary_scores, const Tensor& projected_hidden,
                                    const Tensor& anchors, const Tensor& predecessor_codebook,
                                    const Tensor& successor_codebook, const Tensor& base_positions,
                                    const SamplingConfig* configs, Tensor& drafts,
                                    Tensor& proposal_q, const SelectorWorkspace& workspace,
                                    cudaStream_t stream) {
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
                          candidate_ids.ne[1]};
    if (route == SelectorRoute::Direct) {
        selector_walk_kernel<<<candidate_ids.ne[2], 512, 0, stream>>>(args);
    } else {
        auto* edges = static_cast<float*>(workspace.edges.data);
        selector_lattice_kernel<<<dim3(args.steps * candidate_ids.ne[2], kCandidates), 512, 0,
                                  stream>>>(args, edges);
        CUDA_CHECK(cudaGetLastError());
        selector_lattice_walk_kernel<<<candidate_ids.ne[2], 32, 0, stream>>>(args, edges);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
