#include "launch.h"

#include "ops/common/cuda/pipeline.cuh"

#include <algorithm>
#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace ninfer::ops::markov {
namespace {

using Bf16 = __nv_bfloat16;

struct Problem {
    const Bf16* state;
    const int* tokens;
    const float* context;
    const Bf16* embed;
    const Bf16* weights;
    const int* live;
    Bf16* output;
    Bf16* latent;
    int rank, batch;
};

__device__ __forceinline__ float round_bf16(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float dot_pair(unsigned a, unsigned b, float dot) {
    const auto lo = [](unsigned bits) { return __uint_as_float(bits << 16); };
    const auto hi = [](unsigned bits) { return __uint_as_float(bits & 0xffff0000U); };
    return fmaf(hi(a), hi(b), fmaf(lo(a), lo(b), dot));
}

__device__ __forceinline__ float dot_vector(uint4 a, uint4 b, float dot) {
    dot = dot_pair(a.x, b.x, dot);
    dot = dot_pair(a.y, b.y, dot);
    dot = dot_pair(a.z, b.z, dot);
    return dot_pair(a.w, b.w, dot);
}

__device__ __forceinline__ void publish(const Problem& p, std::int64_t row, int c, int rank,
                                        Bf16 previous, float gate, float candidate, float readout) {
    const auto* ctx    = p.context + row * 3 + c;
    const float g      = round_bf16(__fdiv_rn(1.0f, 1.0f + expf(-round_bf16(gate + ctx[0]))));
    const float value  = round_bf16(tanhf(round_bf16(candidate + ctx[rank])));
    const float keep   = round_bf16(g * __bfloat162float(previous));
    const float update = round_bf16(round_bf16(1.0f - g) * value);
    p.output[row + c]  = __float2bfloat16_rn(keep + update);
    p.latent[row + c]  = __float2bfloat16_rn(tanhf(round_bf16(readout + ctx[2 * rank])));
}

__device__ __forceinline__ void zero(const Problem& p, std::int64_t row, int c) {
    p.output[row + c] = __float2bfloat16_rn(0.0f);
    p.latent[row + c] = __float2bfloat16_rn(0.0f);
}

template <int Group>
__device__ __forceinline__ void reduce(float& g, float& c, float& z) {
#pragma unroll
    for (int offset = Group / 2; offset > 0; offset /= 2) {
        g += __shfl_down_sync(0xffffffff, g, offset, Group);
        c += __shfl_down_sync(0xffffffff, c, offset, Group);
        z += __shfl_down_sync(0xffffffff, z, offset, Group);
    }
}

// Two independent channel warps overlap the recurrent weight loads with the input gather.
template <bool Masked>
__global__ void step_256(const __grid_constant__ Problem p) {
    constexpr int kRank = 256, kWarps = 2;
    constexpr int kPlane = 2 * kRank * kRank;
    __shared__ __align__(16) Bf16 input[2 * kRank];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int c = blockIdx.x * kWarps + warp;
    const int k = lane * 8;
    for (std::int64_t b = blockIdx.y; b < p.batch; b += gridDim.y) {
        const auto row = b * kRank;
        if constexpr (Masked) {
            if (p.live[b] == 0) {
                if (lane == 0) zero(p, row, c);
                continue;
            }
        }

        const auto* src = warp == 0 ? p.state + row : p.embed + std::int64_t(p.tokens[b]) * kRank;
        common::cp_async<16, Cache::cg>(input + tid * 8, src + k);
        common::cp_commit();

        const auto* w = p.weights + c * 2 * kRank + k;
        const auto g0 = common::load_ldg<uint4>(w);
        const auto g1 = common::load_ldg<uint4>(w + kRank);
        const auto c0 = common::load_ldg<uint4>(w + kPlane);
        const auto c1 = common::load_ldg<uint4>(w + kPlane + kRank);
        const auto z0 = common::load_ldg<uint4>(w + 2 * kPlane);
        const auto z1 = common::load_ldg<uint4>(w + 2 * kPlane + kRank);

        common::cp_wait<0>();
        __syncthreads();
        const auto s    = common::load_vec<uint4>(input + k);
        const auto e    = common::load_vec<uint4>(input + kRank + k);
        float gate      = dot_vector(e, g1, dot_vector(s, g0, 0));
        float candidate = dot_vector(e, c1, dot_vector(s, c0, 0));
        float readout   = dot_vector(e, z1, dot_vector(s, z0, 0));
        reduce<32>(gate, candidate, readout);

        if (lane == 0) publish(p, row, c, kRank, input[c], gate, candidate, readout);
        if (b + gridDim.y < p.batch) __syncthreads();
    }
}

template <int Threads, int Group, int FixedRank, bool InPlace, bool Masked, int Cluster = 1>
__global__ void step_grouped(const __grid_constant__ Problem p) {
    extern __shared__ __align__(16) Bf16 staged[];
    constexpr int kChannels = Threads / Group;
    const int rank          = FixedRank ? FixedRank : p.rank;
    const int lane = threadIdx.x % Group, group = threadIdx.x / Group;
    constexpr bool whole_row = InPlace && Cluster == 1;
    const int first          = whole_row ? 0 : blockIdx.x * kChannels;
    const int end            = whole_row ? rank : min(rank, first + kChannels);
    const auto plane         = 2LL * rank * rank;
    for (std::int64_t b = blockIdx.y; b < p.batch; b += gridDim.y) {
        const auto row = b * rank;
        if constexpr (Masked) {
            if (p.live[b] == 0) {
                for (int c = first + threadIdx.x; c < end; c += Threads) zero(p, row, c);
                continue;
            }
        }

        const auto* state = p.state + row;
        const auto* embed = p.embed + std::int64_t(p.tokens[b]) * rank;
        if constexpr (InPlace) {
            // Finish the old-state snapshot across every block that can publish this row.
            for (int k = threadIdx.x * 8; k < rank; k += Threads * 8) {
                common::store_vec(staged + k, common::load_vec<uint4>(state + k));
                common::store_vec(staged + rank + k, common::load_vec<uint4>(embed + k));
            }
            if constexpr (Cluster > 1)
                cooperative_groups::this_cluster().sync();
            else
                __syncthreads();
            state = staged;
            embed = staged + rank;
        }

        for (int c = first + group; c < end; c += kChannels) {
            const auto* weight = p.weights + std::int64_t(c) * 2 * rank;
            float gate = 0.0f, candidate = 0.0f, readout = 0.0f;
            for (int k = lane * 8; k < 2 * rank; k += Group * 8) {
                const auto x = common::load_vec<uint4>(k < rank ? state + k : embed + k - rank);
                gate         = dot_vector(x, common::load_ldg<uint4>(weight + k), gate);
                candidate = dot_vector(x, common::load_ldg<uint4>(weight + plane + k), candidate);
                readout   = dot_vector(x, common::load_ldg<uint4>(weight + 2 * plane + k), readout);
            }
            reduce<Group>(gate, candidate, readout);
            if (lane == 0) publish(p, row, c, rank, state[c], gate, candidate, readout);
        }
        if constexpr (InPlace)
            if (b + gridDim.y < p.batch) __syncthreads();
    }
}

// Warps for different batch rows share the two channels' weights within the block.
template <int Rows, bool Masked>
__global__ void step_256_batched(const __grid_constant__ Problem p) {
    constexpr int rank = 256, plane = 2 * rank * rank, channels = 2;
    constexpr int threads = Rows * channels * 32;
    __shared__ __align__(16) Bf16 weights[channels * 6 * rank];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int c0 = blockIdx.x * channels;
    const int k  = lane * 8;
    for (int i = tid; i < channels * 6 * 32; i += threads) {
        const int ch = i / (6 * 32), part = i / 32 % 6;
        const auto* src = p.weights + (c0 + ch) * 2 * rank + (part / 2) * plane +
                          (part % 2) * rank + (i % 32) * 8;
        common::cp_async<16, Cache::cg>(weights + i * 8, src);
    }
    common::cp_commit();
    common::cp_wait<0>();
    __syncthreads();

    const int channel = warp % channels;
    const int c       = c0 + channel;
    const auto* w     = weights + channel * 6 * rank + k;
    for (std::int64_t b = blockIdx.y * Rows + warp / channels; b < p.batch; b += gridDim.y * Rows) {
        const auto row = b * rank;
        if constexpr (Masked) {
            if (p.live[b] == 0) {
                if (lane == 0) zero(p, row, c);
                continue;
            }
        }
        const auto* state   = p.state + row;
        const auto* embed   = p.embed + std::int64_t(p.tokens[b]) * rank;
        const auto s        = common::load_vec<uint4>(state + k);
        const auto e        = common::load_vec<uint4>(embed + k);
        const auto contract = [&](int offset) {
            return dot_vector(e, common::load_vec<uint4>(w + offset + rank),
                              dot_vector(s, common::load_vec<uint4>(w + offset), 0));
        };
        float gate      = contract(0);
        float candidate = contract(2 * rank);
        float readout   = contract(4 * rank);
        reduce<32>(gate, candidate, readout);
        if (lane == 0) publish(p, row, c, rank, state[c], gate, candidate, readout);
    }
}

template <int Cluster, bool Masked>
void step_in_place_256(Problem p, int rows, cudaStream_t stream) {
    constexpr int threads = 1024 / Cluster;
    cudaLaunchAttribute cluster{};
    cluster.id             = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim = {Cluster, 1, 1};
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(Cluster, rows);
    config.blockDim         = dim3(threads);
    config.dynamicSmemBytes = 4 * 256;
    config.stream           = stream;
    config.attrs            = &cluster;
    config.numAttrs         = 1;
    CUDA_CHECK(
        cudaLaunchKernelEx(&config, step_grouped<threads, 4, 256, true, Masked, Cluster>, p));
}

constexpr int kLargeStepThreads   = 256;
constexpr int kSharedSnapshotRank = 22 * 1024;

// Keep the prior state in shared memory and use registers for the part that exceeds it.
// Each channel uses the entire CTA to contract the snapshot.
template <bool Masked>
__global__
__launch_bounds__(kLargeStepThreads) void step_large_in_place(const __grid_constant__ Problem p) {
    constexpr int kThreads = kLargeStepThreads;
    constexpr int kSpan    = kThreads * 8;
    constexpr int kChunks  = (kMarkovRnnStepMaxInPlaceRank - kSharedSnapshotRank) / kSpan;
    static_assert((kMarkovRnnStepMaxInPlaceRank - kSharedSnapshotRank) % kSpan == 0);
    extern __shared__ __align__(16) Bf16 shared_state[];
    __shared__ float3 warp_sum[kThreads / 32];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int rank        = p.rank;
    const int shared_rank = min(rank, kSharedSnapshotRank);
    const auto plane      = 2LL * rank * rank;
    for (std::int64_t b = blockIdx.y; b < p.batch; b += gridDim.y) {
        const auto row = b * rank;
        if constexpr (Masked) {
            if (p.live[b] == 0) {
                for (int c = tid; c < rank; c += kThreads) zero(p, row, c);
                continue;
            }
        }

        const auto* embed = p.embed + std::int64_t(p.tokens[b]) * rank;
        for (int k = tid * 8; k < shared_rank; k += kSpan)
            common::store_vec(shared_state + k, common::load_vec<uint4>(p.state + row + k));
        uint4 snapshot[kChunks]{};
#pragma unroll
        for (int i = 0; i < kChunks; ++i) {
            const int k = kSharedSnapshotRank + i * kSpan + tid * 8;
            if (k < rank) snapshot[i] = common::load_vec<uint4>(p.state + row + k);
        }
        __syncthreads();

        for (int c = 0; c < rank; ++c) {
            const auto* w = p.weights + std::int64_t(c) * 2 * rank;
            float g = 0.0f, z = 0.0f, value = 0.0f;
#pragma unroll 1
            for (int k = tid * 8; k < shared_rank; k += kSpan) {
                const auto s = common::load_vec<uint4>(shared_state + k);
                g            = dot_vector(s, common::load_ldg<uint4>(w + k), g);
                value        = dot_vector(s, common::load_ldg<uint4>(w + plane + k), value);
                z            = dot_vector(s, common::load_ldg<uint4>(w + 2 * plane + k), z);
            }
#pragma unroll
            for (int i = 0; i < kChunks; ++i) {
                const int k = kSharedSnapshotRank + i * kSpan + tid * 8;
                if (k < rank) {
                    const auto s = snapshot[i];
                    g            = dot_vector(s, common::load_ldg<uint4>(w + k), g);
                    value        = dot_vector(s, common::load_ldg<uint4>(w + plane + k), value);
                    z            = dot_vector(s, common::load_ldg<uint4>(w + 2 * plane + k), z);
                }
            }
#pragma unroll 1
            for (int k = tid * 8; k < rank; k += kSpan) {
                const auto e = common::load_vec<uint4>(embed + k);
                g            = dot_vector(e, common::load_ldg<uint4>(w + rank + k), g);
                value        = dot_vector(e, common::load_ldg<uint4>(w + plane + rank + k), value);
                z            = dot_vector(e, common::load_ldg<uint4>(w + 2 * plane + rank + k), z);
            }
            reduce<32>(g, value, z);
            if (lane == 0) warp_sum[warp] = make_float3(g, value, z);
            __syncthreads();
            if (warp == 0) {
                const auto part = lane < kThreads / 32 ? warp_sum[lane] : make_float3(0, 0, 0);
                g               = part.x;
                value           = part.y;
                z               = part.z;
                reduce<32>(g, value, z);
                if (lane == 0) publish(p, row, c, rank, p.state[row + c], g, value, z);
            }
            __syncthreads();
        }
    }
}
} // namespace

void launch_step(const Tensor& state, const Tensor& tokens, const Tensor& context,
                 const Tensor& embed, const Weight& weights, const Tensor& live, Tensor& output,
                 Tensor& latent, CudaExecutionView execution) {
    const Problem p{static_cast<const Bf16*>(state.data),
                    static_cast<const int*>(tokens.data),
                    static_cast<const float*>(context.data),
                    static_cast<const Bf16*>(embed.data),
                    static_cast<const Bf16*>(weights.qdata),
                    static_cast<const int*>(live.data),
                    static_cast<Bf16*>(output.data),
                    static_cast<Bf16*>(latent.data),
                    state.ne[0],
                    state.ne[1]};
    const auto launch = [&]<bool Masked>() {
        const auto stream = execution.stream();
        const int batch   = std::min(p.batch, 65535);
        if (output.data != state.data) {
            if (p.rank == 256) {
                if (p.batch >= 4) {
                    constexpr int rows = 4;
                    const dim3 grid(p.rank / 2, std::min(65535, 1 + (p.batch - 1) / rows));
                    step_256_batched<rows, Masked><<<grid, 64 * rows, 0, stream>>>(p);
                } else {
                    step_256<Masked><<<dim3(p.rank / 2, batch), 64, 0, stream>>>(p);
                }
            } else {
                const dim3 grid(p.rank / 2, batch);
                step_grouped<64, 32, 0, false, Masked><<<grid, 64, 0, stream>>>(p);
            }
        } else {
            const dim3 grid(1, batch);
            constexpr int kSharedRank = 12 * 1024;
            if (p.rank == 256)
                step_in_place_256<4, Masked>(p, batch, stream);
            else if (p.rank <= kSharedRank)
                step_grouped<1024, 4, 0, true, Masked><<<grid, 1024, 4 * p.rank, stream>>>(p);
            else {
                const auto shared = 2 * std::min(p.rank, kSharedSnapshotRank);
                step_large_in_place<Masked><<<grid, kLargeStepThreads, shared, stream>>>(p);
            }
        }
    };
    if (p.live)
        launch.template operator()<true>();
    else
        launch.template operator()<false>();
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::markov
