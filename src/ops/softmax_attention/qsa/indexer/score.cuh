#pragma once

#include "ops/softmax_attention/qsa/indexer/work_map.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::qsa {

__device__ __forceinline__ int indexer_swizzle(int row, int d) { return d ^ ((row & 7) << 3); }

template <int QueryTile, bool VectorCache>
__global__ __launch_bounds__((QueryTile > 16 ? QueryTile * 8 : 128)) void indexer_score_mma_kernel(
    const __grid_constant__ IndexerArguments a) {
    static_assert(QueryTile == 4 || QueryTile == 8 || QueryTile == 16 || QueryTile == 32);
    constexpr int Threads   = QueryTile > 16 ? QueryTile * 8 : 128;
    constexpr int KeyGroups = QueryTile > 16 ? 1 : 16 / QueryTile;
    constexpr int WarpKeys  = kIndexerKeyTile / KeyGroups;
    constexpr int NTiles    = WarpKeys / 8;
    __shared__ IndexerJobs jobs;
    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    auto* qs = shared;
    auto* ks = qs + QueryTile * 4 * 128;
    if (!jobs.build<QueryTile, kIndexerKeyTile>(a)) return;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int per_row  = (a.width + QueryTile - 1) / QueryTile;
    int previous_group = -1;
    for (int job = blockIdx.x; job < jobs.prefix[jobs.groups]; job += gridDim.x) {
        const int group = jobs.group_for(job), b = group / per_row;
        const int q0     = (group % per_row) * QueryTile;
        const int g0     = (job - jobs.prefix[group]) * kIndexerKeyTile;
        const int length = jobs.tile_visible(a, b, q0, QueryTile);
        for (int chunk = tid; group != previous_group && chunk < QueryTile * 4 * 16;
             chunk += Threads) {
            const int row = chunk / 16, d = (chunk % 16) * 8;
            const int t = q0 + row / 4, slot = b * a.width + t;
            auto* dst = qs + row * 128 + indexer_swizzle(row, d);
            if (t < a.width && jobs.visible[slot] > kIndexerSelected) {
                const auto* src = a.query + a.column(slot) * 512 + (row % 4) * 128 + d;
                common::cp_async<16, Cache::cg>(dst, src);
            } else
                common::store_vec(dst, make_int4(0, 0, 0, 0));
        }
        previous_group = group;
        for (int chunk = tid; chunk < kIndexerKeyTile * 16; chunk += Threads) {
            const int row = chunk / 16, d = (chunk % 16) * 8;
            auto* dst = ks + row * 128 + indexer_swizzle(row, d);
            if (g0 + row < length) {
                const auto* src = a.key(b, g0 + row) + d;
                if constexpr (VectorCache)
                    common::cp_async<16, Cache::cg>(dst, src);
                else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) dst[j] = src[j];
                }
            } else
                common::store_vec(dst, make_int4(0, 0, 0, 0));
        }
        common::cp_commit();
        common::cp_wait<0>();
        __syncthreads();
        float acc[NTiles][4]{};
        const int qr = (warp / KeyGroups) * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int qc = (lane >> 4) * 8;
        const int kr = (warp % KeyGroups) * WarpKeys + (lane & 7), kc = ((lane >> 3) & 1) * 8;
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            unsigned af[4];
            common::ldmatrix_x4(
                af[0], af[1], af[2], af[3],
                common::smem_addr(qs + qr * 128 + indexer_swizzle(qr, step * 16 + qc)));
#pragma unroll
            for (int nt = 0; nt < NTiles; ++nt) {
                unsigned bf[2];
                const int row = kr + nt * 8;
                common::ldmatrix_x2(
                    bf[0], bf[1],
                    common::smem_addr(ks + row * 128 + indexer_swizzle(row, step * 16 + kc)));
                common::mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], af[0], af[1],
                                 af[2], af[3], bf[0], bf[1]);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NTiles; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float value = fmaxf(acc[nt][j], 0.0F);
                value += __shfl_xor_sync(0xffffffffu, value, 4);
                value += __shfl_xor_sync(0xffffffffu, value, 8);
                const int row = (warp / KeyGroups) * 16 + lane / 4 + (j / 2) * 8;
                const int t = q0 + row / 4, slot = b * a.width + t;
                const int g = g0 + (warp % KeyGroups) * WarpKeys + nt * 8 + (lane % 4) * 2 + j % 2;
                if (row % 4 == 0 && t < a.width && jobs.visible[slot] > kIndexerSelected &&
                    g < jobs.visible[slot])
                    a.scores[std::int64_t(slot) * a.columns + g] = value;
            }
        __syncthreads();
    }
}
} // namespace ninfer::ops::detail::qsa
