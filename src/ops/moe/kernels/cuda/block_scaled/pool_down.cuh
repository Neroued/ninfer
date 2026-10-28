#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/host/launch.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled {

// A small active expert set shares one output tile. Each CTA owns the sum across
// that set, including zero contributions for tokens that do not select an expert.
template <int TopK, int Columns, class Contraction>
struct PoolDownStorage {
    typename Contraction::Tile tile;
    int experts[2 * TopK], count;
    float coefficients[Columns];
};

template <int Hidden, int Experts, int TopK, int Columns, class Contraction, class Group>
__global__ __launch_bounds__(Contraction::threads) void pool_down(
    Contraction contraction, Group groups, const int* ids, const float* coefficients, float* output,
    int tokens, int minimum_experts, int expert_limit, int row_window) {
    const int active = groups.load_range[2];
    if (active <= minimum_experts || active > expert_limit) return;
    using Storage = PoolDownStorage<TopK, Columns, Contraction>;
    auto& storage = *reinterpret_cast<Storage*>(common::shared_storage<sizeof(Storage)>());
    if (threadIdx.x == 0) storage.count = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < Experts; e += blockDim.x)
        if (groups.offsets[e] < groups.offsets[e + 1]) {
            const int slot        = atomicAdd(&storage.count, 1);
            storage.experts[slot] = e;
        }
    __syncthreads();

    constexpr int row_tiles    = Hidden / 128;
    constexpr int warp_columns = Columns >= 64 ? Columns / 2 : Columns;
    constexpr int atoms        = warp_columns / 8;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp % 4, wn = warp / 4;
    const int column_tiles = (tokens + Columns - 1) / Columns;
    const int window       = row_window ? min(row_window, row_tiles) : row_tiles;
    const int span         = column_tiles * window;
    const int tasks        = ((row_tiles + window - 1) / window) * span;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int row0 = (task / span * window + task % window) * 128;
        if (row0 >= Hidden) continue;
        const int begin = task % span / window * Columns;
        float total[2][atoms][4]{};
        for (int slot = 0; slot < active; ++slot) {
            const int expert = storage.experts[slot];
            if (threadIdx.x < Columns) {
                const int token = begin + threadIdx.x;
                int assignment  = -1;
                if (token < tokens)
                    for (int j = 0; j < TopK; ++j)
                        if (ids[token * TopK + j] == expert) assignment = token * TopK + j;
                storage.tile.sources[threadIdx.x] =
                    assignment >= 0 ? groups.inverse[assignment] : -1;
                storage.coefficients[threadIdx.x] = assignment >= 0 ? coefficients[assignment] : 0;
            }
            __syncthreads();
            float partial[2][atoms][4]{};
            const float factor = contraction(storage.tile, row0, expert, begin, tokens, partial);
#pragma unroll
            for (int n = 0; n < atoms; ++n) {
                const int column = wn * warp_columns + n * 8 + (lane & 3) * 2;
#pragma unroll
                for (int m = 0; m < 2; ++m)
#pragma unroll
                    for (int v = 0; v < 4; ++v)
                        total[m][n][v] +=
                            partial[m][n][v] * factor * storage.coefficients[column + (v & 1)];
            }
            __syncthreads();
        }
#pragma unroll
        for (int n = 0; n < atoms; ++n) {
            const int column = begin + wn * warp_columns + n * 8 + (lane & 3) * 2;
#pragma unroll
            for (int m = 0; m < 2; ++m)
#pragma unroll
                for (int v = 0; v < 4; ++v)
                    if (column + (v & 1) < tokens) {
                        const int row = row0 + wm * 32 + m * 16 + (lane >> 2) + (v / 2) * 8;
                        output[std::int64_t(column + (v & 1)) * Hidden + row] = total[m][n][v];
                    }
        }
        __syncthreads();
    }
}

template <int Hidden, int Experts, int TopK, int Columns, class Contraction, class Group>
void launch_pool_down(Contraction contraction, Group groups, const int* ids,
                      const float* coefficients, float* output, int tokens, int minimum_experts,
                      int expert_limit, int row_window, int queued_ctas_per_sm,
                      CudaExecutionView execution) {
    if (!output || expert_limit > 2 * TopK || queued_ctas_per_sm <= 0)
        throw std::logic_error("invalid reduced expert pool");
    using Storage         = PoolDownStorage<TopK, Columns, Contraction>;
    constexpr auto kernel = pool_down<Hidden, Experts, TopK, Columns, Contraction, Group>;
    const int bytes       = common::prepare_shared<sizeof(Storage), kernel>();
    const int tiles       = (Hidden / 128) * ((tokens + Columns - 1) / Columns);
    const int blocks      = std::min(tiles, execution.multiprocessor_count() * queued_ctas_per_sm);
    kernel<<<blocks, Contraction::threads, bytes, execution.stream()>>>(
        contraction, groups, ids, coefficients, output, tokens, minimum_experts, expert_limit,
        row_window);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
