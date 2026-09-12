#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear_topk/grouped_ksplit_topk.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int Capacity, class Schedule>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void w8_grouped_ksplit_topk_kernel(
    const __nv_bfloat16* __restrict__ hidden, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, std::int32_t valid_rows,
    std::uint64_t* __restrict__ partial_keys, std::int32_t producer_groups, int columns) {
    using Geometry           = W8VocabularyProjectionGeometry;
    constexpr int kHidden    = Geometry::kInputRows;
    constexpr int kTileK     = Schedule::kTileKPerWarp;
    constexpr int kWarps     = Schedule::kKWarps;
    constexpr int kRows      = Schedule::kRowsPerCta;
    constexpr int kGroupK    = Schedule::kGroupK;
    constexpr int kGroups    = kHidden / kGroupK;
    constexpr int kTileCols  = Schedule::kTileTokens;
    constexpr int kTokenMmas = kTileCols / 8;
    constexpr int kRowTiles  = kLinearTopKGroupedRows / kRows;
    constexpr unsigned kMask = 0xffffffffu;
    static_assert(kRows == 16 && kLinearTopKGroupedRows == 128);
    static_assert((kHidden % kGroupK) == 0);
    static_assert(Capacity <= kTileCols);

    union SharedStorage {
        struct {
            std::uint8_t codes[kRows][kGroupK];
            __nv_bfloat16 activations[kWarps][kTileCols * kTileK];
            std::uint8_t scales[kRows][Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared
                                           ? Schedule::kScaleBytesPerRow
                                           : 1];
        } staging;

        float partial[kWarps * kTokenMmas * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    __shared__ GroupedKSplitTopKStorage<Capacity, kWarps> topk;
    auto& code_shared  = shared.staging.codes;
    auto& x_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;
    const int b_row   = lane & 7;
    const int b_koff  = ((lane >> 3) & 1) << 3;
    const int warp_k0 = k_split * kTileK;

    grouped_ksplit_topk_initialize(topk);

    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
        const int row_begin =
            static_cast<int>(blockIdx.x) * kLinearTopKGroupedRows + row_tile * kRows;

        const auto stage_activation = [&](int group_k0) {
            constexpr int kItems = Capacity * (kTileK / 8);
            for (int item = lane; item < kItems; item += 32) {
                const int column = item / (kTileK / 8);
                const int k8     = item - column * (kTileK / 8);
                const int source = column < columns ? column : 0;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    &x_shared[warp][column * kTileK + w8_small_t_swizzle_64(column, k8 * 8)],
                    hidden + static_cast<std::int64_t>(source) * kHidden + group_k0 +
                        warp * kTileK + k8 * 8, column < columns ? 16 : 0);
            }
        };

        const auto stage_weight = [&](int group_k0) {
#pragma unroll
            for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
                const int local_row = warp * Schedule::kRowsPerLoaderWarp + row_item;
                const int row       = row_begin + local_row;
                for (int chunk = lane; chunk < kGroupK / 16; chunk += 32) {
                    const int swizzled_chunk = chunk ^ (local_row & 7);
                    cp_async<16, Schedule::kWeightCache>(
                        &code_shared[local_row][swizzled_chunk * 16],
                        codes + static_cast<std::int64_t>(row) * kHidden + group_k0 + chunk * 16);
                }
            }
            if constexpr (Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared) {
                constexpr int kScaleChunksPerRow = Schedule::kScaleBytesPerRow / 16;
                for (int item = tid; item < kRows * kScaleChunksPerRow;
                     item += Schedule::kThreads) {
                    const int local_row = item / kScaleChunksPerRow;
                    const int chunk     = item - local_row * kScaleChunksPerRow;
                    const int row       = row_begin + local_row;
                    cp_async<16, Schedule::kWeightCache>(
                        &scale_shared[local_row][chunk * 16],
                        scales + (static_cast<std::int64_t>(row) * Geometry::kGroupsPerRow +
                                  group_k0 / 32 + chunk * 8) *
                                     2);
                }
            }
        };

        float accumulators[kTokenMmas][4] = {};
        stage_weight(0);
        stage_activation(0);
        cp_commit();
        cp_wait<0>();
        __syncthreads();

#pragma unroll
        for (int group_index = 0; group_index < kGroups; ++group_index) {
            const int group_k0       = group_index * kGroupK;
            unsigned lane_scale_pair = 0;
            if (lid < 2) {
                if constexpr (Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared) {
                    lane_scale_pair = *reinterpret_cast<const unsigned*>(
                        &scale_shared[gid + lid * 8][warp_k0 / 16]);
                } else {
                    lane_scale_pair = *reinterpret_cast<const unsigned*>(
                        scales + (static_cast<std::int64_t>(row_begin + gid + lid * 8) *
                                      Geometry::kGroupsPerRow +
                                  group_k0 / 32 + warp_k0 / 32) *
                                     2);
                }
            }
            const unsigned top_scale_pair = __shfl_sync(kMask, lane_scale_pair, lane & ~3);
            const unsigned bot_scale_pair = __shfl_sync(kMask, lane_scale_pair, (lane & ~3) + 1);

#pragma unroll
            for (int quant_group = 0; quant_group < 2; ++quant_group) {
                float group_acc[kTokenMmas][4] = {};
#pragma unroll
                for (int k_pair = 0; k_pair < 2; ++k_pair) {
                    const int k_step     = quant_group * 2 + k_pair;
                    const int code_col   = k_step * 16 + lid * 2;
                    const auto load_pair = [&](int local_row, int col) {
                        const int chunk  = (warp_k0 + col) >> 4;
                        const int offset = (chunk ^ (local_row & 7)) * 16 + (col & 15);
                        return static_cast<unsigned>(*reinterpret_cast<const std::uint16_t*>(
                            &code_shared[local_row][offset]));
                    };
                    const unsigned a0 = w8_small_t_bf16_pair_from_s8(load_pair(gid, code_col));
                    const unsigned a1 = w8_small_t_bf16_pair_from_s8(load_pair(gid + 8, code_col));
                    const unsigned a2 = w8_small_t_bf16_pair_from_s8(load_pair(gid, code_col + 8));
                    const unsigned a3 =
                        w8_small_t_bf16_pair_from_s8(load_pair(gid + 8, code_col + 8));
#pragma unroll
                    for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                        unsigned b0;
                        unsigned b1;
                        const int row = token_mma * 8 + b_row;
                        ldmatrix_x2(
                            b0, b1,
                            smem_addr(
                                &x_shared[k_split][row * kTileK + w8_small_t_swizzle_64(
                                                                      row, k_step * 16 + b_koff)]));
                        mma_bf16(group_acc[token_mma][0], group_acc[token_mma][1],
                                 group_acc[token_mma][2], group_acc[token_mma][3], a0, a1, a2, a3,
                                 b0, b1);
                    }
                }
                const unsigned top_bits =
                    quant_group == 0 ? top_scale_pair & 0xffffu : top_scale_pair >> 16;
                const unsigned bot_bits =
                    quant_group == 0 ? bot_scale_pair & 0xffffu : bot_scale_pair >> 16;
                const float top_scale = __half2float(__ushort_as_half(top_bits));
                const float bot_scale = __half2float(__ushort_as_half(bot_bits));
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    accumulators[token_mma][0] =
                        fmaf(group_acc[token_mma][0], top_scale, accumulators[token_mma][0]);
                    accumulators[token_mma][1] =
                        fmaf(group_acc[token_mma][1], top_scale, accumulators[token_mma][1]);
                    accumulators[token_mma][2] =
                        fmaf(group_acc[token_mma][2], bot_scale, accumulators[token_mma][2]);
                    accumulators[token_mma][3] =
                        fmaf(group_acc[token_mma][3], bot_scale, accumulators[token_mma][3]);
                }
            }

            if (group_index + 1 < kGroups) {
                __syncthreads();
                stage_weight(group_k0 + kGroupK);
                stage_activation(group_k0 + kGroupK);
                cp_commit();
                cp_wait<0>();
                __syncthreads();
            }
        }

        __syncthreads();
        float* partial = shared.partial;
        if ((k_split & 1) != 0) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                store_vec(partial + ((k_split * kTokenMmas + token_mma) * 32 + lane) * 4,
                          make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                      accumulators[token_mma][2], accumulators[token_mma][3]));
            }
        }
        __syncthreads();

        if ((k_split & 1) == 0) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const float4 partner = load_vec<float4>(
                    partial + (((k_split + 1) * kTokenMmas + token_mma) * 32 + lane) * 4);
                accumulators[token_mma][0] += partner.x;
                accumulators[token_mma][1] += partner.y;
                accumulators[token_mma][2] += partner.z;
                accumulators[token_mma][3] += partner.w;
                if (k_split != 0) {
                    store_vec(partial + ((k_split * kTokenMmas + token_mma) * 32 + lane) * 4,
                              make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                          accumulators[token_mma][2], accumulators[token_mma][3]));
                }
            }
        }
        __syncthreads();

        if (k_split == 0) {
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                float4 sum = make_float4(accumulators[token_mma][0], accumulators[token_mma][1],
                                         accumulators[token_mma][2], accumulators[token_mma][3]);
#pragma unroll
                for (int split = 2; split < kWarps; split += 2) {
                    const float4 value = load_vec<float4>(
                        partial + ((split * kTokenMmas + token_mma) * 32 + lane) * 4);
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                const int column0 = token_mma * 8 + 2 * lid;
                if (column0 < Capacity) {
                    partial[gid * kTileCols + column0]       = sum.x;
                    partial[(gid + 8) * kTileCols + column0] = sum.z;
                }
                if (column0 + 1 < Capacity) {
                    partial[gid * kTileCols + column0 + 1]       = sum.y;
                    partial[(gid + 8) * kTileCols + column0 + 1] = sum.w;
                }
            }
        }
        __syncthreads();
        grouped_ksplit_topk_consume<Capacity, kTileCols, kWarps>(partial, topk,
                                                                             row_begin, valid_rows, columns);
    }

    grouped_ksplit_topk_publish(topk, partial_keys, producer_groups, columns);
}

using Launch = void (*)(const Tensor&, const Weight&, std::int32_t, const LinearTopKWorkspace&,
                        cudaStream_t);

template <int Capacity>
void launch_tile(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                  const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    using Schedule = W8SmallTMmaSchedule<8, Capacity, 2, W8SmallTMmaScaleAccess::Shared>;
    w8_grouped_ksplit_topk_kernel<Capacity, Schedule>
        <<<workspace.producer_groups, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const std::uint8_t*>(head.qdata),
            static_cast<const std::uint8_t*>(head.scales), valid_rows,
            static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups, hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Indices>
constexpr auto make_launchers(std::index_sequence<Indices...>) {
    return std::array<Launch, sizeof...(Indices)>{&launch_tile<8 * (1 + Indices)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<3>{});

} // namespace

void linear_topk_w8_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                           const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    if (workspace.tile_columns == 0) {
        kLaunchers[(hidden.ne[1] - 1) / 8](hidden, head, valid_rows, workspace, stream);
    } else {
        linear_topk_w8_m64_launch(hidden, head, valid_rows, workspace, stream);
    }
}
} // namespace ninfer::ops::detail
