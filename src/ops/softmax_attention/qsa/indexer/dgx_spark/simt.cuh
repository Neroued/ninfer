#pragma once
#include "ops/softmax_attention/qsa/indexer/work_map.cuh"
#include "ops/common/cuda/memory.cuh"

namespace ninfer::ops::detail::qsa {
template <int Lanes, int Threads, bool Vector, class Arguments, class Jobs>
__device__ __forceinline__ void indexer_score_simt(const Arguments& a, int worker, int stride,
                                                   Jobs& jobs) {
    if (!jobs.template build<1, kIndexerKeyTile>(a)) return;
    constexpr int D = 128 / Lanes, Groups = Threads / Lanes;
    const int lane = threadIdx.x % Lanes, group = threadIdx.x / Lanes;
    const unsigned mask = (0xffffffffu >> (32 - Lanes)) << ((threadIdx.x % 32) / Lanes * Lanes);
    float q[4][D];
    int previous = -1;
    for (int job = worker; job < jobs.prefix[jobs.groups]; job += stride) {
        const int slot = jobs.group_for(job);
        const int b    = indexer_segment(a, slot, Arguments::ragged ? 0 : slot / a.width);
        const int g0 = (job - jobs.prefix[slot]) * kIndexerKeyTile;
        if (slot != previous) {
#pragma unroll
            for (int h = 0; h < 4; ++h)
#pragma unroll
                for (int i = 0; i < D; ++i)
                    q[h][i] =
                        __bfloat162float(a.query[a.column(slot) * 512 + h * 128 + lane * D + i]);
            previous = slot;
        }
        for (int g = g0 + group; g < min(g0 + kIndexerKeyTile, jobs.visible[slot]); g += Groups) {
            const auto* key = a.key(b, g) + lane * D;
            float k[D];
            if constexpr (Vector) {
#pragma unroll
                for (int i = 0; i < D; i += 4) {
                    const auto bits   = common::load_vec<int2>(key + i);
                    const auto* words = reinterpret_cast<const __nv_bfloat16*>(&bits);
#pragma unroll
                    for (int j = 0; j < 4; ++j) k[i + j] = __bfloat162float(words[j]);
                }
            } else {
#pragma unroll
                for (int i = 0; i < D; ++i) k[i] = __bfloat162float(key[i]);
            }
            float value = 0;
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                float dot = 0;
#pragma unroll
                for (int i = 0; i < D; ++i) dot = fmaf(q[h][i], k[i], dot);
#pragma unroll
                for (int offset = Lanes / 2; offset; offset /= 2)
                    dot += __shfl_down_sync(mask, dot, offset, Lanes);
                value += fmaxf(dot, 0.0f);
            }
            if (lane == 0) a.scores[std::int64_t(slot) * a.columns + g] = value;
        }
    }
}

template <int Lanes, int Threads, bool Vector, int MaxQueries = 64, bool BatchedLoop = false,
          class Arguments = IndexerArguments>
__global__
__launch_bounds__(Threads, (!BatchedLoop && Lanes == 16) ? 8 : 1) void indexer_score_simt_kernel(
    const __grid_constant__ Arguments a) {
    __shared__ std::conditional_t<Arguments::ragged, RaggedIndexerSlots<MaxQueries>,
                                  IndexerJobs<MaxQueries>>
        jobs;
    for_each_indexer_job_group<MaxQueries, BatchedLoop>(
        a, [&](const auto& group, int worker, int stride) {
            indexer_score_simt<Lanes, Threads, Vector>(group, worker, stride, jobs);
        });
}
} // namespace ninfer::ops::detail::qsa
