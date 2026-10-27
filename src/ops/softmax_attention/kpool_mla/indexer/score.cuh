#pragma once

#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"

namespace ninfer::ops::detail::kpool {

// Query groups are interleaved at each key tile. A persistent worker retains its query
// fragments while nearby groups can reuse that tile through L2.
template <int Queries, int Keys, int Stages>
__global__
__launch_bounds__((Queries < 4 ? 4 : Queries) * 32,
                  4) void indexer_score_kernel(const __grid_constant__ IndexerArguments a) {
    constexpr int KeyGroups = Queries < 4 ? 4 / Queries : 1;
    constexpr int Threads   = Queries * KeyGroups * 32;
    constexpr int WarpKeys = Keys / KeyGroups, MT = WarpKeys / 16;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    for (int slot = blockIdx.x; slot < a.width * a.batch; slot += gridDim.x) {
        const int end = a.end(slot), count = end / 4;
        const auto column = a.column(slot);
        if (tid == 0) {
            a.ends[column]   = end;
            a.counts[column] = min(count, kIndexSelected);
        }
        if (count <= kIndexSelected)
            for (int i = tid; i < count; i += Threads) a.ids[column * kIndexSelected + i] = i;
    }
    if (a.columns <= kIndexSelected) return;
    const int per_batch = (a.width + Queries - 1) / Queries;
    const int groups = per_batch * a.batch, group = blockIdx.x % groups;
    const int b = group / per_batch, first = group % per_batch * Queries;
    // Bound positions directly by the resolved request and this CTA's query group.
    const int live_end        = min(min(a.width, a.valid[b] - a.begin), first + Queries);
    const auto request_column = std::int64_t(b) * a.tokens + a.begin;
    const int visible = live_end > first ? (a.positions[request_column + live_end - 1] + 1) / 4 : 0;
    const int start   = blockIdx.x / groups * Keys;
    if (visible <= kIndexSelected || start >= visible) return;
    const int step = gridDim.x / groups * Keys;
    const int t = first + (Queries == 1 ? 0 : warp / KeyGroups), slot = b * a.width + t;
    const auto column = request_column + t;
    const int limit   = Queries == 1 ? visible : t < live_end ? (a.positions[column] + 1) / 4 : 0;
    // MMA rows are pools and columns are heads. Four adjacent lanes reduce the 32 heads.
    unsigned qf[4][8][2];
    float weight[4][2];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
#pragma unroll
        for (int j = 0; j < 2; ++j)
            weight[n][j] = limit > kIndexSelected
                               ? a.coefficients[column * 32 + n * 8 + (lane % 4) * 2 + j]
                               : 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                int h = n * 8 + lane / 4, d = k * 16 + (lane % 4) * 2 + j * 8;
                qf[n][k][j] = limit > kIndexSelected ? __ldg(reinterpret_cast<const unsigned*>(
                                                           a.query + column * 4096 + h * 128 + d))
                                                     : 0;
            }
        }
    }
    extern __shared__ __align__(16) __nv_bfloat16 pipe[];
    const auto* table = a.tables + std::int64_t(__ldg(a.table_rows + b)) * a.table_stride;
    const auto policy = common::l2_streaming_policy();
    // Issue the next page offsets before consuming the current tile.
    long long offsets[Keys / 16];
    const auto addresses = [&](int g0) {
#pragma unroll
        for (int p = 0; p < Keys / 16; ++p) {
            if (g0 + p * 16 < visible) {
                const auto* pointer = table + g0 / 16 + p;
                asm volatile("ld.global.nc.u64 %0, [%1];" : "=l"(offsets[p]) : "l"(pointer));
            } else
                offsets[p] = 0;
        }
    };
    // Each copy covers eight pooled rows at 128 threads. Their XOR row phase repeats,
    // and enumerating chunks makes the prefetched page index a compile-time constant.
    constexpr int CopiesPerPage = 256 / Threads;
    static_assert(Threads % 128 == 0 && 256 % Threads == 0);
    const int copy_row      = tid / 16;
    auto* copy_destination  = pipe + copy_row * 128 + ((tid % 16 * 8) ^ ((copy_row & 7) << 3));
    const auto* copy_source = reinterpret_cast<const char*>(a.keys) + tid * 16;
    const auto stage        = [&](int g0, int buffer) {
        auto* dst = copy_destination + buffer * Keys * 128;
#pragma unroll
        for (int chunk = 0; chunk < Keys * 16 / Threads; ++chunk) {
            const auto* in = copy_source + offsets[chunk / CopiesPerPage] +
                             (chunk % CopiesPerPage) * Threads * 16;
            const int row = copy_row + chunk * (Threads / 16);
            common::cp_async_cache_hint<Cache::cg>(dst + chunk * Threads * 8, in,
                                                   g0 + row < visible ? 16 : 0, policy);
        }
        common::cp_commit();
    };
    // Only the buffer and K-step offsets change while retaining these lane fragments.
    const unsigned key_lane_base =
        common::smem_addr(pipe) +
        (warp % KeyGroups * WarpKeys + (lane & 15)) * 128 * sizeof(__nv_bfloat16);
    const unsigned key_column_xor = ((lane >> 4) ^ (lane & 7)) << 4;
    addresses(start);
    if constexpr (Stages == 2) {
        stage(start, 0);
        addresses(start + step);
    }
    for (int g0 = start, iteration = 0; g0 < visible; g0 += step, ++iteration) {
        const int buffer = Stages == 2 ? iteration % 2 : 0;
        if constexpr (Stages == 1) stage(g0, buffer);
        common::cp_wait<0>();
        __syncthreads();
        if constexpr (Stages == 2)
            if (g0 + step < visible) stage(g0 + step, 1 - buffer);
        addresses(g0 + Stages * step);
        float acc[MT][4][4]{};
#pragma unroll
        for (int k = 0; k < 8; ++k) {
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                unsigned af[4];
                common::ldmatrix_x4(af[0], af[1], af[2], af[3],
                                    key_lane_base + (buffer * Keys + m * 16) * 256 +
                                        ((k * 32) ^ key_column_xor));
#pragma unroll
                for (int n = 0; n < 4; ++n)
                    common::mma_bf16(acc[m][n][0], acc[m][n][1], acc[m][n][2], acc[m][n][3], af[0],
                                     af[1], af[2], af[3], qf[n][k][0], qf[n][k][1]);
            }
        }
#pragma unroll
        for (int m = 0; m < MT; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                float v0 = 0, v1 = 0;
#pragma unroll
                for (int n = 0; n < 4; ++n) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        if (n < 2)
                            v0 += weight[n][j] * fmaxf(acc[m][n][2 * half + j], 0.f);
                        else
                            v1 += weight[n][j] * fmaxf(acc[m][n][2 * half + j], 0.f);
                    }
                }
                float value = v0 + v1;
                value += __shfl_xor_sync(0xffffffffu, value, 1);
                value += __shfl_xor_sync(0xffffffffu, value, 2);
                const int g = g0 + warp % KeyGroups * WarpKeys + m * 16 + lane / 4 + half * 8;
                if (lane % 4 == 0 && limit > kIndexSelected && g < limit)
                    a.scores[std::int64_t(slot) * a.columns + g] = value;
            }
        }
        // With two stages, the next wait/barrier protects the old buffer before reuse.
        if constexpr (Stages == 1) __syncthreads();
    }
}
} // namespace ninfer::ops::detail::kpool
