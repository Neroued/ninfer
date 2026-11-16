#pragma once

#include "ops/softmax_attention/qsa/indexer/dgx_spark/score.cuh"

#include <cuda.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::qsa {

// storage_bytes starts at a.keys; each referenced page offset must be 128-byte aligned.
inline CUtensorMap make_indexer_key_map(const IndexerArguments& a, std::size_t storage_bytes) {
    CUtensorMap map{};
    const cuuint64_t dimensions[]{64, 16, storage_bytes / 128};
    const cuuint64_t strides[]{cuuint64_t(a.key_row_stride * 2), 128};
    const unsigned box[]{64, 16, 2}, step[]{1, 1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3,
        static_cast<void*>(const_cast<__nv_bfloat16*>(a.keys)), dimensions, strides, box, step,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("QSA indexer TMA descriptor: ") +
                                 (name ? name : "CUDA error"));
    }
    return map;
}

struct IndexerTmaStorage {
    static constexpr int Stages = 2;
    __nv_bfloat16 keys[Stages * 64 * 128];
    std::uint64_t ready[Stages];
    ScoreWorkMap<kIndexerMaxWindowQueries / 4> jobs;
};

inline constexpr std::size_t kIndexerTmaSharedBytes = sizeof(IndexerTmaStorage);

__device__ __forceinline__ int indexer_tma_key_offset(int row, int d) {
    const int logical = row / 16 * 2048 + d / 64 * 1024 + row % 16 * 64 + d % 64;
    return logical ^ (((logical / 64) & 7) * 8);
}

__global__
__launch_bounds__(128) void indexer_score_tma_kernel(const __grid_constant__ IndexerArguments a,
                                                     const __grid_constant__ CUtensorMap map) {
    constexpr int Stages = IndexerTmaStorage::Stages;
    constexpr int Tile = 4, Threads = 128, KeyGroups = 4, WarpKeys = 16, NT = 2;
    if (a.columns <= kIndexerSelected) return;
    extern __shared__ __align__(1024) unsigned char shared[];
    auto& s    = *reinterpret_cast<IndexerTmaStorage*>(shared);
    auto& jobs = s.jobs;
    if (!jobs.template build<Tile>(a)) return;
    const int total = jobs.prefix[jobs.groups];
    if (blockIdx.x >= total) return;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int per_row = (a.width + Tile - 1) / Tile;
    if (tid == 0) {
#pragma unroll
        for (int buffer = 0; buffer < Stages; ++buffer)
            common::cta_mbarrier_init(&s.ready[buffer], 1);
        common::cta_mbarrier_fence_init();
    }
    __syncthreads();
    const auto stage = [&](int job, int buffer) {
        const int group = jobs.group_for(job), b = group / per_row;
        const int first = (job - jobs.prefix[group]) * 64;
        const int rows = min(64, jobs.length[group] - first), full_pages = rows / 16;
        auto* tile = s.keys + buffer * 64 * 128;
        if (tid == 0) {
            common::cta_mbarrier_arrive_expect_tx(&s.ready[buffer], full_pages * 4096);
            const auto* table = a.tables + std::int64_t(a.table_rows[b]) * a.table_stride;
#pragma unroll
            for (int page = 0; page < 4; ++page) {
                if (page < full_pages) {
                    const auto offset = __ldg(table + first / 16 + page);
                    common::tma_load_3d(tile + page * 2048, &map, 0, 0, int(offset / 128),
                                        &s.ready[buffer]);
                }
            }
        }
        if (rows < 64) {
            const int tail_first = full_pages * 16;
            for (int i = tid; i < (64 - tail_first) * 16; i += Threads) {
                const int row = tail_first + i / 16, d = i % 16 * 8;
                auto* out = tile + indexer_tma_key_offset(row, d);
                if (row < rows)
                    common::cp_async<16, Cache::cg>(out, a.key(b, first + row) + d);
                else
                    common::store_vec(out, make_int4(0, 0, 0, 0));
            }
        }
        // Only the final partial page uses cp.async; it is outside TMA byte accounting.
        if (rows % 16) common::cp_commit();
    };
    stage(blockIdx.x, 0);
    unsigned qf[8][4];
    int previous = -1, limit[2], slot[2];
    for (int job = blockIdx.x, iteration = 0; job < total; job += gridDim.x, ++iteration) {
        const int group = jobs.group_for(job), b = group / per_row;
        const int q0 = group % per_row * Tile, first = (job - jobs.prefix[group]) * 64;
        const int buffer = iteration % Stages;
        if (group != previous) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int row = warp / KeyGroups * 16 + lane / 4 + half * 8;
                const int t   = q0 + row / 4;
                slot[half]    = b * a.width + t;
                limit[half]   = t < a.width ? a.visible(slot[half]) : 0;
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
        if (tid == 0) common::cta_mbarrier_wait(&s.ready[buffer], (iteration / Stages) & 1);
        if (min(64, jobs.length[group] - first) % 16) common::cp_wait<0>();
        __syncthreads();
        const int next = job + gridDim.x;
        if (next < total) stage(next, 1 - buffer);
        auto* tile = s.keys + buffer * 64 * 128;
        float acc[NT][4]{};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                unsigned bf[2];
                const int row = warp % KeyGroups * WarpKeys + n * 8 + (lane & 7);
                const int d   = k * 16 + ((lane >> 3) & 1) * 8;
                common::ldmatrix_x2(bf[0], bf[1],
                                    common::smem_addr(tile + indexer_tma_key_offset(row, d)));
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
                const int g = first + warp % KeyGroups * WarpKeys + n * 8 + (lane % 4) * 2 + j % 2;
                if ((lane / 4) % 4 == 0 && limit[j / 2] > kIndexerSelected && g < limit[j / 2])
                    a.scores[std::int64_t(slot[j / 2]) * a.columns + g] = v;
            }
        }
        // Publish generic reads and tail writes before a later TMA reuses the buffer.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
    }
}

} // namespace ninfer::ops::detail::qsa
