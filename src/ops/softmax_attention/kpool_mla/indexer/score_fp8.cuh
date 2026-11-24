#pragma once

#include "ops/softmax_attention/kpool_mla/indexer/arguments.h"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/quant/fp8.cuh"
#include "ops/softmax_attention/kpool_mla/indexer/work.cuh"

namespace ninfer::ops::detail::kpool {

template <int Queries, int Keys, int Stages, int MinBlocks, int WarpQueries = 1,
          bool EpiloguePrefetch = false>
__global__ __launch_bounds__(
    (Queries / WarpQueries < 4 ? 4 : Queries / WarpQueries) * 32,
    MinBlocks) void indexer_score_fp8_kernel(const __grid_constant__ Fp8IndexerArguments f) {
    const auto& a           = f.index;
    constexpr int KeyGroups = Queries / WarpQueries < 4 ? 4 / (Queries / WarpQueries) : 1;
    constexpr int Threads   = Queries / WarpQueries * KeyGroups * 32;
    constexpr int WarpKeys = Keys / KeyGroups, MT = WarpKeys / 16;
    constexpr int CopyGroups = Threads / 128, PageChunks = Keys / (16 * CopyGroups);
    static_assert(Threads % 128 == 0 && Keys % (16 * CopyGroups) == 0);
    static_assert(Stages >= 1 && Stages <= 3);
    static_assert(!EpiloguePrefetch || Stages == 1);
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
    auto work = score_work<Queries, Keys>(a);
    if (work.batch < 0) return;
    // These fields are warp-uniform. REDUX preserves their bits and makes that
    // uniformity visible to ptxas, keeping the hot address arithmetic in URs.
    work.batch    = __reduce_or_sync(0xffffffffu, unsigned(work.batch));
    work.first    = __reduce_or_sync(0xffffffffu, unsigned(work.first));
    work.start    = __reduce_or_sync(0xffffffffu, unsigned(work.start));
    work.step     = __reduce_or_sync(0xffffffffu, unsigned(work.step));
    work.live     = __reduce_or_sync(0xffffffffu, unsigned(work.live));
    work.position = __reduce_or_sync(0xffffffffu, unsigned(work.position));
    const int b = work.batch, first = work.first, start = work.start, step = work.step;
    const int live_end = min(work.live, first + Queries), visible = (work.position + live_end) / 4;
    if (start >= visible) return;
    const int t = first + warp / KeyGroups * WarpQueries, slot = b * a.width + t;
    int limit[WarpQueries];
    unsigned qf[WarpQueries][4][4][2];
    float weight[WarpQueries][4][2];
#pragma unroll
    for (int q = 0; q < WarpQueries; ++q) {
        limit[q]           = t + q < live_end ? (work.position + t + q + 1) / 4 : 0;
        const int prepared = b * f.query_width + a.begin + t + q - f.query_begin;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
#pragma unroll
            for (int j = 0; j < 2; ++j)
                weight[q][n][j] = limit[q] > kIndexSelected
                                      ? f.coefficients[prepared * 32 + n * 8 + (lane % 4) * 2 + j]
                                      : 0.F;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int h = n * 8 + lane / 4, d = k * 32 + (lane % 4) * 4 + j * 16;
                    qf[q][n][k][j] = limit[q] > kIndexSelected
                                         ? __ldg(reinterpret_cast<const unsigned*>(
                                               f.query + prepared * 4096 + h * 128 + d))
                                         : 0;
                }
            }
        }
    }
    extern __shared__ __align__(16) std::uint8_t fp8_pipe[];
    auto* scale_pipe     = fp8_pipe + Stages * Keys * 128;
    const auto* table    = a.tables + std::int64_t(__ldg(a.table_rows + b)) * a.table_stride;
    const auto policy    = common::l2_streaming_policy();
    const int copy_group = tid / 128, copy_row = tid / 8;
    long long offsets[PageChunks];
    const auto addresses = [&](int g0) {
#pragma unroll
        for (int p = 0; p < PageChunks; ++p) {
            const int page = p * CopyGroups + copy_group;
            if (g0 + page * 16 < visible) {
                const auto* pointer = table + g0 / 16 + page;
                asm volatile("ld.global.nc.u64 %0, [%1];" : "=l"(offsets[p]) : "l"(pointer));
            } else
                offsets[p] = 0;
        }
    };
    // Each 128-thread group copies one complete page; its page-array index is static.
    auto* copy_destination  = fp8_pipe + copy_row * 128 + ((tid % 8 * 16) ^ ((copy_row & 7) << 4));
    const auto* copy_source = f.keys + (tid % 128) * 16;
    const auto stage        = [&](int g0, int buffer) {
#pragma unroll
        for (int p = 0; p < PageChunks; ++p) {
            const int row = copy_row + p * (Threads / 8);
            common::cp_async_cache_hint<Cache::cg>(
                copy_destination + buffer * Keys * 128 + p * Threads * 16, copy_source + offsets[p],
                g0 + row < visible ? 16 : 0, policy);
            if (tid % 128 == 0) {
                const int first = (p * CopyGroups + copy_group) * 16;
                common::cp_async_cache_hint<Cache::cg>(
                    scale_pipe + buffer * Keys + first, f.scales + offsets[p],
                    min(16, max(0, visible - g0 - first)), policy);
            }
        }
        common::cp_commit();
    };
    const unsigned key_lane_base =
        common::smem_addr(fp8_pipe) + (warp % KeyGroups * WarpKeys + (lane & 15)) * 128;
    const unsigned key_column_xor = ((lane >> 4) ^ (lane & 7)) << 4;
    if constexpr (Stages <= 2) addresses(start);
    if constexpr (Stages == 2 || EpiloguePrefetch) {
        stage(start, 0);
        addresses(start + step);
    } else if constexpr (Stages > 2) {
#pragma unroll
        for (int s = 0; s < Stages - 1; ++s) {
            if (start + s * step < visible) {
                addresses(start + s * step);
                stage(start + s * step, s);
            } else {
                // Empty groups preserve wait distance when the tail has fewer tiles.
                common::cp_commit();
            }
        }
        addresses(start + (Stages - 1) * step);
    }
    for (int g0 = start, iteration = 0; g0 < visible; g0 += step, ++iteration) {
        const int buffer = Stages == 1 ? 0 : iteration % Stages;
        if constexpr (Stages == 1 && !EpiloguePrefetch) stage(g0, buffer);
        common::cp_wait<(Stages > 2 ? Stages - 2 : 0)>();
        __syncthreads();
        if constexpr (Stages == 2)
            if (g0 + step < visible) stage(g0 + step, 1 - buffer);
        if constexpr (Stages > 2) {
            const int next = g0 + (Stages - 1) * step;
            if (next < visible)
                stage(next, (iteration + Stages - 1) % Stages);
            else
                common::cp_commit();
        }
        if constexpr (!EpiloguePrefetch) addresses(g0 + Stages * step);
        float acc[WarpQueries][MT][4][4]{};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
#pragma unroll
            for (int m = 0; m < MT; ++m) {
                unsigned af[4];
                common::ldmatrix_x4(af[0], af[1], af[2], af[3],
                                    key_lane_base + (buffer * Keys + m * 16) * 128 +
                                        ((k * 32) ^ key_column_xor));
#pragma unroll
                for (int q = 0; q < WarpQueries; ++q) {
#pragma unroll
                    for (int n = 0; n < 4; ++n)
                        common::mma_fp8_e4m3(acc[q][m][n][0], acc[q][m][n][1], acc[q][m][n][2],
                                             acc[q][m][n][3], af[0], af[1], af[2], af[3],
                                             qf[q][n][k][0], qf[q][n][k][1]);
                }
            }
        }
        unsigned saved_scales[MT][2];
        if constexpr (EpiloguePrefetch) {
#pragma unroll
            for (int m = 0; m < MT; ++m) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int row = warp % KeyGroups * WarpKeys + m * 16 + lane / 4 + half * 8;
                    saved_scales[m][half] = scale_pipe[row];
                }
            }
            // K and its scales are no longer read from shared memory. Reuse the
            // single buffer while the current tile's FP32 epilogue runs.
            __syncthreads();
            if (g0 + step < visible) stage(g0 + step, 0);
            addresses(g0 + 2 * step);
        }
#pragma unroll
        for (int q = 0; q < WarpQueries; ++q) {
#pragma unroll
            for (int m = 0; m < MT; ++m) {
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    float v0 = 0.F, v1 = 0.F;
#pragma unroll
                    for (int n = 0; n < 4; ++n) {
#pragma unroll
                        for (int j = 0; j < 2; ++j) {
                            if (n < 2)
                                v0 += weight[q][n][j] * fmaxf(acc[q][m][n][2 * half + j], 0.F);
                            else
                                v1 += weight[q][n][j] * fmaxf(acc[q][m][n][2 * half + j], 0.F);
                        }
                    }
                    float value = v0 + v1;
                    value += __shfl_xor_sync(0xffffffffu, value, 1);
                    value += __shfl_xor_sync(0xffffffffu, value, 2);
                    const int row = warp % KeyGroups * WarpKeys + m * 16 + lane / 4 + half * 8;
                    const int g   = g0 + row;
                    if (lane % 4 == 0 && limit[q] > kIndexSelected && g < limit[q])
                        a.scores[std::int64_t(slot + q) * a.columns + g] =
                            value * common::ue8m0_to_float(EpiloguePrefetch
                                                               ? saved_scales[m][half]
                                                               : scale_pipe[buffer * Keys + row]);
                }
            }
        }
        if constexpr (Stages == 1 && !EpiloguePrefetch) __syncthreads();
    }
}
} // namespace ninfer::ops::detail::kpool
