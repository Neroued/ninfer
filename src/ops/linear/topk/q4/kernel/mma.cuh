#pragma once
#include "ops/common/numeric/convert.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/cuda/pipeline.cuh"
#include "ops/common/cuda/mma.cuh"
#include "ops/common/cuda/memory.cuh"
#include "core/weight.h"
#include "ops/linear/topk/dispatch.h"

#include "core/device.h"
#include "ops/linear/topk/workspace.h"

#include <cub/warp/warp_merge_sort.cuh>
#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroupsPerRow = kLinearTopKHidden / 64;
constexpr int kCodeRowBytes = kLinearTopKHidden / 2;

using MmaWarpSort              = cub::WarpMergeSort<std::uint64_t, 2, 32>;
constexpr int kRowsPerProducer = 64;

template <int TileColumns, int kBlockK = 128>
struct MmaSchedule {
    static constexpr int kColumnWarps  = TileColumns == 120 ? 3 : 2;
    static constexpr int kWarps        = 4 * kColumnWarps;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kBlockColumns = TileColumns;
};

template <int TileColumns, int kBlockK = 128>
struct alignas(16) Q4MmaMainloopStorage {
    static constexpr int kBlockRows    = 64;
    static constexpr int kBlockColumns = MmaSchedule<TileColumns, kBlockK>::kBlockColumns;
    __nv_bfloat16 weights[kBlockRows][kBlockK];
    __nv_bfloat16 activations[kBlockColumns][kBlockK];
    std::uint8_t codes[kBlockRows][kBlockK / 2];
    std::uint16_t scales[kBlockRows][kBlockK / 64];
};

template <int TileColumns, int kBlockK = 128>
struct Q4MmaReductionStorage {
    static constexpr int kBlockRows = 64;
    float scores[TileColumns][kBlockRows];
    typename MmaWarpSort::TempStorage sort[MmaSchedule<TileColumns, kBlockK>::kWarps];
};

template <int TileColumns, int kBlockK = 128>
union alignas(16) Q4MmaReusableStorage {
    Q4MmaMainloopStorage<TileColumns, kBlockK> mainloop;
    Q4MmaReductionStorage<TileColumns, kBlockK> reduction;
};

template <int TileColumns, int kBlockK = 128>
__global__
__launch_bounds__(MmaSchedule<TileColumns, kBlockK>::kThreads, 2) void q4_mma_linear_topk_kernel(
    const __nv_bfloat16* __restrict__ hidden, const std::uint8_t* __restrict__ weight_codes,
    const std::uint8_t* __restrict__ weight_scales,
    const std::int32_t* __restrict__ row_to_global_ids, std::uint64_t* __restrict__ partial_keys,
    std::int32_t producer_groups, std::int32_t columns) {
    constexpr int kKTiles       = kLinearTopKHidden / kBlockK;
    constexpr int kBlockRows    = 64;
    constexpr int kSortItems    = 2;
    constexpr int kWarps        = MmaSchedule<TileColumns, kBlockK>::kWarps;
    constexpr int kThreads      = MmaSchedule<TileColumns, kBlockK>::kThreads;
    constexpr int kColumnWarps  = MmaSchedule<TileColumns, kBlockK>::kColumnWarps;
    constexpr int kBlockColumns = Q4MmaMainloopStorage<TileColumns, kBlockK>::kBlockColumns;
    constexpr int kWarpColumns  = kBlockColumns / kColumnWarps;
    constexpr int kTokenMmas    = kWarpColumns / 8;
    static_assert(TileColumns % 8 == 0 && TileColumns <= 128);
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto& reusable = *reinterpret_cast<Q4MmaReusableStorage<TileColumns, kBlockK>*>(shared_bytes);
    const int column_begin = static_cast<int>(blockIdx.y) * TileColumns;
    const int live_columns = min(TileColumns, columns - column_begin);
    hidden += static_cast<std::int64_t>(column_begin) * kLinearTopKHidden;

    auto& mainloop = reusable.mainloop;

    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int warp_row = warp / kColumnWarps;
    const int warp_col = warp - warp_row * kColumnWarps;
    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_matrix = lane >> 3;
    const int a_rowoff = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_coloff = (a_matrix >> 1) << 3;
    const int b_row    = lane & 7;
    const int b_coloff = ((lane >> 3) & 1) << 3;

    __syncthreads();

    const int row_begin = static_cast<int>(blockIdx.x) * kRowsPerProducer;
    {
        float accumulators[kTokenMmas][4] = {};

        const auto stage_activation = [&](int k_tile) {
            const int k_begin    = k_tile * kBlockK;
            constexpr int kItems = kBlockColumns * (kBlockK / 8);
            for (int item = tid; item < kItems; item += kThreads) {
                const int column = item / (kBlockK / 8);
                const int k8     = item - column * (kBlockK / 8);
                auto* destination =
                    &mainloop.activations[column][common::swizzle_xor_64(column, k8 * 8)];
                if (column < live_columns) {
                    common::cp_async<16, Cache::ca>(
                        destination, hidden +
                                         static_cast<std::int64_t>(column) * kLinearTopKHidden +
                                         k_begin + k8 * 8);
                } else {
                    common::cp_async_zfill<16, Cache::ca>(destination, hidden + k_begin + k8 * 8,
                                                          0);
                }
            }
        };

        const auto stage_weight = [&](int k_tile) {
            const int k_begin     = k_tile * kBlockK;
            constexpr int kChunks = kBlockRows * (kBlockK / 2 / 16);
            for (int item = tid; item < kChunks; item += kThreads) {
                const int local_row = item / (kBlockK / 2 / 16);
                const int chunk     = item - local_row * (kBlockK / 2 / 16);
                common::cp_async<16, Cache::cg>(
                    &mainloop.codes[local_row][chunk * 16],
                    weight_codes +
                        static_cast<std::int64_t>(row_begin + local_row) * kCodeRowBytes +
                        k_begin / 2 + chunk * 16);
            }
            for (int local_row = tid; local_row < kBlockRows; local_row += kThreads) {
                const std::int64_t group =
                    static_cast<std::int64_t>(row_begin + local_row) * kGroupsPerRow + k_begin / 64;
                if constexpr (kBlockK == 64)
                    mainloop.scales[local_row][0] =
                        reinterpret_cast<const std::uint16_t*>(weight_scales)[group];
                else
                    common::cp_async<4>(&mainloop.scales[local_row][0],
                                        weight_scales + group * sizeof(std::uint16_t));
            }
        };

        const auto decode_weight = [&]() {
            constexpr int kChunksPerRow = kBlockK / 8;
            for (int item = tid; item < kBlockRows * kChunksPerRow; item += kThreads) {
                const int row     = item / kChunksPerRow;
                const int chunk   = item - row * kChunksPerRow;
                const int col     = chunk * 8;
                const float scale = __half2float(__ushort_as_half(mainloop.scales[row][col / 64]));
                const unsigned packed =
                    *reinterpret_cast<const unsigned*>(&mainloop.codes[row][col / 2]);
                common::Bf16x8Bits decoded;
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const unsigned byte = packed >> (pair * 8);
                    const int q0        = (static_cast<int>(byte & 0x0fu) ^ 0x08) - 0x08;
                    const int q1        = (static_cast<int>((byte >> 4) & 0x0fu) ^ 0x08) - 0x08;
                    decoded.pair[pair]  = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                                static_cast<float>(q1) * scale);
                }
                common::store_vec(&mainloop.weights[row][common::swizzle_xor_64(row, col)],
                                  decoded.raw);
            }
        };

        stage_activation(0);
        stage_weight(0);
        common::cp_commit();

#pragma unroll 1
        for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
            common::cp_wait<0>();
            __syncthreads();
            decode_weight();
            __syncthreads();

            const int next = k_tile + 1;
            if (next < kKTiles) {
                stage_weight(next);
                common::cp_commit();
            }

            const auto load_fragments = [&](int k_step, unsigned(&a)[4],
                                            unsigned(&b)[kTokenMmas][2]) {
                const int weight_row = warp_row * 16 + a_rowoff;
                const int weight_col = k_step * 16 + a_coloff;
                common::ldmatrix_x4(
                    a[0], a[1], a[2], a[3],
                    common::smem_addr(&mainloop.weights[weight_row][common::swizzle_xor_64(
                        weight_row, weight_col)]));
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    const int activation_row = warp_col * kWarpColumns + token_mma * 8 + b_row;
                    const int activation_col = k_step * 16 + b_coloff;
                    common::ldmatrix_x2(
                        b[token_mma][0], b[token_mma][1],
                        common::smem_addr(
                            &mainloop.activations[activation_row][common::swizzle_xor_64(
                                activation_row, activation_col)]));
                }
            };

            unsigned a_fragments[2][4];
            unsigned b_fragments[2][kTokenMmas][2];
            load_fragments(0, a_fragments[0], b_fragments[0]);
#pragma unroll
            for (int k_step = 0; k_step < kBlockK / 16; ++k_step) {
                const int slot = k_step & 1;
                if (k_step + 1 < kBlockK / 16) {
                    load_fragments(k_step + 1, a_fragments[slot ^ 1], b_fragments[slot ^ 1]);
                }
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    common::mma_bf16(accumulators[token_mma][0], accumulators[token_mma][1],
                                     accumulators[token_mma][2], accumulators[token_mma][3],
                                     a_fragments[slot][0], a_fragments[slot][1],
                                     a_fragments[slot][2], a_fragments[slot][3],
                                     b_fragments[slot][token_mma][0],
                                     b_fragments[slot][token_mma][1]);
                }
            }

            if (next < kKTiles) {
                __syncthreads();
                stage_activation(next);
                common::cp_commit();
            }
        }

        __syncthreads();
        auto& scores         = reusable.reduction.scores;
        const int local_row0 = warp_row * 16 + gid;
        const int local_row1 = local_row0 + 8;
#pragma unroll
        for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
            const int column0 = warp_col * kWarpColumns + token_mma * 8 + 2 * lid;
            if (column0 < TileColumns) {
                scores[column0][local_row0] = accumulators[token_mma][0];
                scores[column0][local_row1] = accumulators[token_mma][2];
            }
            if (column0 + 1 < TileColumns) {
                scores[column0 + 1][local_row0] = accumulators[token_mma][1];
                scores[column0 + 1][local_row1] = accumulators[token_mma][3];
            }
        }
    }
    __syncthreads();

    const int reducer_warp = tid >> 5;
    const int reducer_lane = tid & 31;
    for (int column = reducer_warp; column < live_columns; column += kWarps) {
        std::uint64_t keys[kSortItems] = {};
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int local_row = reducer_lane * 2 + item;
            const int row       = row_begin + local_row;
            if (row < kLinearTopKMappedRows)
                keys[item] = common::score_id_order_key(
                    reusable.reduction.scores[column][local_row], row_to_global_ids[row]);
        }
        MmaWarpSort(reusable.reduction.sort[reducer_warp])
            .Sort(keys, common::ScoreIdOrderGreater{});
#pragma unroll
        for (int item = 0; item < kSortItems; ++item) {
            const int rank = reducer_lane * kSortItems + item;
            if (rank < kLinearTopK) {
                if (column < live_columns) {
                    const auto dst =
                        (static_cast<std::int64_t>(column_begin + column) * producer_groups +
                         blockIdx.x) *
                            kLinearTopK +
                        rank;
                    partial_keys[dst] = keys[item];
                }
            }
        }
    }
    __syncthreads();
}

} // namespace
} // namespace ninfer::ops::detail
