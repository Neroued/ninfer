#pragma once
#include "ops/softmax_attention/qsa/indexer/arguments.h"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::qsa {
struct ScoreWorkMap {
    static constexpr int capacity = kIndexerMaxWindowQueries / 4;
    int length[capacity], prefix[capacity + 1], groups;

    template <int Tile>
    __device__ bool build(const IndexerArguments& a) {
        const int tid = threadIdx.x, per_row = (a.width + Tile - 1) / Tile,
                  count = per_row * a.batch;
        int visible     = 0;
        if (tid < count) {
            const int b = tid / per_row, first = tid % per_row * Tile;
            const int end = min(min(a.width, a.valid[b] - a.begin), first + Tile);
            visible       = end > first
                                ? (a.positions[a.begin + end - 1 + std::int64_t(b) * a.tokens] + 1) / 4
                                : 0;
            length[tid]   = visible;
        }
        if (!__syncthreads_or(visible > kIndexerSelected)) return false;
        if (tid < 32) {
            int sum = visible > kIndexerSelected ? (visible + 63) / 64 : 0;
#pragma unroll
            for (int shift = 1; shift < 32; shift *= 2) {
                const int x = __shfl_up_sync(0xffffffffu, sum, shift);
                if (tid >= shift) sum += x;
            }
            if (tid < count) prefix[tid + 1] = sum;
            if (tid == 0) {
                prefix[0] = 0;
                groups    = count;
            }
        }
        __syncthreads();
        return true;
    }

    __device__ int group_for(int job) const {
        int lo = 0, hi = groups;
        while (lo + 1 < hi) {
            const int mid = (lo + hi) / 2;
            if (prefix[mid] <= job)
                lo = mid;
            else
                hi = mid;
        }
        return lo;
    }
};

template <int Tile, bool Vector, int Stages>
__global__ __launch_bounds__(Tile > 16 ? Tile * 8 : 128) void indexer_score_pipeline_kernel(
    const __grid_constant__ IndexerArguments a) {
    constexpr int Threads = Tile > 16 ? Tile * 8 : 128, KeyGroups = Tile > 16 ? 1 : 16 / Tile,
                  WarpKeys = 64 / KeyGroups, NT = WarpKeys / 8;
    if (a.columns <= kIndexerSelected) return;
    static_assert(Tile >= 4 && Tile <= 64 && (Tile & (Tile - 1)) == 0);
    static_assert(Stages == 1 || Stages == 2);
    __shared__ ScoreWorkMap jobs;
    extern __shared__ __align__(16) __nv_bfloat16 pipe_keys[];
    if (!jobs.build<Tile>(a)) return;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32,
              per_row = (a.width + Tile - 1) / Tile;
    const auto stage  = [&](int job, int buffer) {
        const int group = jobs.group_for(job), b = group / per_row,
                  g0 = (job - jobs.prefix[group]) * 64;
        auto* dst    = pipe_keys + buffer * 64 * 128;
        for (int i = tid; i < 64 * 16; i += Threads) {
            const int row = i / 16, d = i % 16 * 8;
            auto* out = dst + row * 128 + (d ^ ((row & 7) << 3));
            if (g0 + row < jobs.length[group]) {
                const auto* in = a.key(b, g0 + row) + d;
                if constexpr (Vector)
                    common::cp_async<16, Cache::cg>(out, in);
                else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) out[j] = in[j];
                }
            } else
                common::store_vec(out, make_int4(0, 0, 0, 0));
        }
        common::cp_commit();
    };
    const int total = jobs.prefix[jobs.groups];
    if (blockIdx.x >= total) return;
    if constexpr (Stages == 2) stage(blockIdx.x, 0);
    unsigned qf[8][4];
    int previous = -1, limit[2], slot[2];
    for (int job = blockIdx.x, step_job = 0; job < total; job += gridDim.x, ++step_job) {
        const int group = jobs.group_for(job), b = group / per_row, q0 = group % per_row * Tile,
                  g0     = (job - jobs.prefix[group]) * 64;
        const int buffer = Stages == 2 ? step_job % 2 : 0;
        if constexpr (Stages == 1) stage(job, buffer);
        if (group != previous) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = warp / KeyGroups * 16 + lane / 4 + half * 8, t = q0 + row / 4;
                slot[half]  = b * a.width + t;
                limit[half] = t < a.width ? a.visible(slot[half]) : 0;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
#pragma unroll
                    for (int col = 0; col < 2; ++col) {
                        const int d = k * 16 + (lane % 4) * 2 + col * 8;
                        qf[k][half + 2 * col] =
                            limit[half] > kIndexerSelected
                                ? __ldg(reinterpret_cast<const unsigned*>(
                                      a.query + a.column(slot[half]) * 512 + row % 4 * 128 + d))
                                : 0;
                    }
                }
            }
            previous = group;
        }
        common::cp_wait<0>();
        __syncthreads();
        if constexpr (Stages == 2)
            if (job + gridDim.x < total) stage(job + gridDim.x, 1 - buffer);
        auto* ks = pipe_keys + buffer * 64 * 128;
        float acc[NT][4]{};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                unsigned bf[2];
                const int row = warp % KeyGroups * WarpKeys + n * 8 + (lane & 7),
                          d   = k * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(bf[0], bf[1],
                                    common::smem_addr(ks + row * 128 + (d ^ ((row & 7) << 3))));
                common::mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], qf[k][0], qf[k][1],
                                 qf[k][2], qf[k][3], bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float v = fmaxf(acc[n][j], 0.f);
                v += __shfl_xor_sync(0xffffffffu, v, 4);
                v += __shfl_xor_sync(0xffffffffu, v, 8);
                const int g = g0 + warp % KeyGroups * WarpKeys + n * 8 + (lane % 4) * 2 + j % 2;
                if ((lane / 4) % 4 == 0 && limit[j / 2] > kIndexerSelected && g < limit[j / 2])
                    a.scores[std::int64_t(slot[j / 2]) * a.columns + g] = v;
            }
        }
        __syncthreads();
    }
}
} // namespace ninfer::ops::detail::qsa
