#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/score_id_order.cuh"
#include "ops/linear/w8/w8_small_t_mma.cuh"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <cub/warp/warp_merge_sort.cuh>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kGroupsPerRow = kLinearTopKHidden / 32;

template <int Rows>
using M64WarpSort = cub::WarpMergeSort<std::uint64_t, Rows == 64 ? 2 : 3, 32>;

template <int TileColumns, int kBlockK = 128, int kRowsPerProducer = 128>
struct M64Schedule {
    static constexpr int kColumnWarps  = TileColumns == 120  ? 3
                                         : TileColumns <= 48 ? TileColumns / 8
                                                             : 2;
    static constexpr int kWarps        = 4 * kColumnWarps;
    static constexpr int kThreads      = kWarps * 32;
    static constexpr int kBlockColumns = TileColumns;
};

union alignas(16) Bf16x8 {
    uint4 raw;
    __nv_bfloat162 pair[4];
};

template <int TileColumns, int kBlockK = 128, int kRowsPerProducer = 128>
struct alignas(16) M64MainloopStorage {
    static constexpr int kBlockRows = 64;
    static constexpr int kBlockColumns =
        M64Schedule<TileColumns, kBlockK, kRowsPerProducer>::kBlockColumns;
    __nv_bfloat16 weights[kBlockRows][kBlockK];
    __nv_bfloat16 activations[kBlockColumns][kBlockK];
    std::uint8_t codes[kBlockRows][kBlockK];
    std::uint16_t scales[kBlockRows][kBlockK / 32];
};

template <int TileColumns, int kBlockK = 128, int kRowsPerProducer = 128>
struct M64ReductionStorage {
    float scores[TileColumns]
                [M64MainloopStorage<TileColumns, kBlockK, kRowsPerProducer>::kBlockRows];
    typename M64WarpSort<kRowsPerProducer>::TempStorage
        sort[M64Schedule<TileColumns, kBlockK, kRowsPerProducer>::kWarps];
};

template <int TileColumns, int kBlockK = 128, int kRowsPerProducer = 128>
union alignas(16) M64ReusableStorage {
    M64MainloopStorage<TileColumns, kBlockK, kRowsPerProducer> mainloop;
    M64ReductionStorage<TileColumns, kBlockK, kRowsPerProducer> reduction;
};

__device__ __forceinline__ int swizzle_128(int row, int column) {
    return (((column >> 3) ^ (row & 7)) << 3) | (column & 7);
}

template <int TileColumns, int kBlockK = 128, int kRowsPerProducer = 128>
__global__
__launch_bounds__(M64Schedule<TileColumns, kBlockK, kRowsPerProducer>::kThreads,
                  2) void w8_m64_linear_topk_kernel(const __nv_bfloat16* __restrict__ hidden,
                                                    const std::uint8_t* __restrict__ weight_codes,
                                                    const std::uint8_t* __restrict__ weight_scales,
                                                    std::int32_t valid_rows,
                                                    std::uint64_t* __restrict__ partial_keys,
                                                    std::int32_t producer_groups,
                                                    std::int32_t columns) {
    constexpr int kKTiles      = kLinearTopKHidden / kBlockK;
    constexpr int kBlockRows   = 64;
    constexpr int kRowTiles    = kRowsPerProducer / kBlockRows;
    constexpr int kSortItems   = kRowsPerProducer == 64 ? 2 : 3;
    constexpr int kWarps       = M64Schedule<TileColumns, kBlockK, kRowsPerProducer>::kWarps;
    constexpr int kThreads     = M64Schedule<TileColumns, kBlockK, kRowsPerProducer>::kThreads;
    constexpr int kColumnWarps = M64Schedule<TileColumns, kBlockK, kRowsPerProducer>::kColumnWarps;
    constexpr int kBlockColumns =
        M64MainloopStorage<TileColumns, kBlockK, kRowsPerProducer>::kBlockColumns;
    constexpr int kWarpColumns = kBlockColumns / kColumnWarps;
    constexpr int kTokenMmas   = kWarpColumns / 8;
    static_assert(kRowsPerProducer == 64 || kRowsPerProducer == 128);
    static_assert(TileColumns % 8 == 0 && TileColumns <= 128);
    extern __shared__ __align__(16) unsigned char shared_bytes[];
    auto& reusable = *reinterpret_cast<M64ReusableStorage<TileColumns, kBlockK, kRowsPerProducer>*>(
        shared_bytes);
    auto* top_keys = reinterpret_cast<std::uint64_t(*)[kLinearTopK]>(
        shared_bytes + sizeof(M64ReusableStorage<TileColumns, kBlockK, kRowsPerProducer>));
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

    if constexpr (kRowsPerProducer > 64) {
        for (int p = tid; p < TileColumns * kLinearTopK; p += kThreads) {
            top_keys[p / kLinearTopK][p % kLinearTopK] = 0;
        }
    }
    __syncthreads();

    for (int row_tile = 0; row_tile < kRowTiles; ++row_tile) {
        const int row_begin =
            static_cast<int>(blockIdx.x) * kRowsPerProducer + row_tile * kBlockRows;
        {
            float accumulators[kTokenMmas][4] = {};

            const auto stage_activation = [&](int k_tile) {
                const int k_begin    = k_tile * kBlockK;
                constexpr int kItems = kBlockColumns * (kBlockK / 8);
                for (int item = tid; item < kItems; item += kThreads) {
                    const int column  = item / (kBlockK / 8);
                    const int k8      = item - column * (kBlockK / 8);
                    auto* destination = &mainloop.activations[column][swizzle_128(column, k8 * 8)];
                    if (column < live_columns) {
                        cp_async<16, Cache::ca>(destination, hidden +
                                                                 static_cast<std::int64_t>(column) *
                                                                     kLinearTopKHidden +
                                                                 k_begin + k8 * 8);
                    } else {
                        cp_async_zfill<16, Cache::ca>(destination, hidden + k_begin + k8 * 8, 0);
                    }
                }
            };

            const auto stage_weight = [&](int k_tile) {
                const int k_begin     = k_tile * kBlockK;
                constexpr int kChunks = kBlockRows * (kBlockK / 16);
                for (int item = tid; item < kChunks; item += kThreads) {
                    const int local_row = item / (kBlockK / 16);
                    const int chunk     = item - local_row * (kBlockK / 16);
                    cp_async<16, Cache::cg>(&mainloop.codes[local_row][chunk * 16],
                                            weight_codes +
                                                static_cast<std::int64_t>(row_begin + local_row) *
                                                    kLinearTopKHidden +
                                                k_begin + chunk * 16);
                }
                for (int local_row = tid; local_row < kBlockRows; local_row += kThreads) {
                    const std::int64_t group =
                        static_cast<std::int64_t>(row_begin + local_row) * kGroupsPerRow +
                        k_begin / 32;
                    cp_async<kBlockK / 32 * sizeof(std::uint16_t)>(
                        &mainloop.scales[local_row][0],
                        weight_scales + group * sizeof(std::uint16_t));
                }
            };

            const auto decode_weight = [&]() {
                constexpr int kChunksPerRow = kBlockK / 8;
                for (int item = tid; item < kBlockRows * kChunksPerRow; item += kThreads) {
                    const int row   = item / kChunksPerRow;
                    const int chunk = item - row * kChunksPerRow;
                    const int col   = chunk * 8;
                    const float scale =
                        __half2float(__ushort_as_half(mainloop.scales[row][col / 32]));
                    const uint2 packed = *reinterpret_cast<const uint2*>(&mainloop.codes[row][col]);
                    Bf16x8 decoded;
#pragma unroll
                    for (int pair = 0; pair < 4; ++pair) {
                        const unsigned word = (pair < 2 ? packed.x : packed.y) >> ((pair & 1) * 16);
                        const int q0 = static_cast<int>(static_cast<std::int8_t>(word & 0xffu));
                        const int q1 =
                            static_cast<int>(static_cast<std::int8_t>((word >> 8) & 0xffu));
                        decoded.pair[pair] = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                                   static_cast<float>(q1) * scale);
                    }
                    store_vec(&mainloop.weights[row][swizzle_128(row, col)], decoded.raw);
                }
            };

            stage_activation(0);
            stage_weight(0);
            cp_commit();

#pragma unroll 1
            for (int k_tile = 0; k_tile < kKTiles; ++k_tile) {
                cp_wait<0>();
                __syncthreads();
                decode_weight();
                __syncthreads();

                const int next = k_tile + 1;
                if (next < kKTiles) {
                    stage_weight(next);
                    cp_commit();
                }

                const auto load_fragments = [&](int k_step, unsigned(&a)[4],
                                                unsigned(&b)[kTokenMmas][2]) {
                    const int weight_row = warp_row * 16 + a_rowoff;
                    const int weight_col = k_step * 16 + a_coloff;
                    ldmatrix_x4(
                        a[0], a[1], a[2], a[3],
                        smem_addr(
                            &mainloop.weights[weight_row][swizzle_128(weight_row, weight_col)]));
#pragma unroll
                    for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                        const int activation_row = warp_col * kWarpColumns + token_mma * 8 + b_row;
                        const int activation_col = k_step * 16 + b_coloff;
                        ldmatrix_x2(b[token_mma][0], b[token_mma][1],
                                    smem_addr(&mainloop.activations[activation_row][swizzle_128(
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
                        mma_bf16(accumulators[token_mma][0], accumulators[token_mma][1],
                                 accumulators[token_mma][2], accumulators[token_mma][3],
                                 a_fragments[slot][0], a_fragments[slot][1], a_fragments[slot][2],
                                 a_fragments[slot][3], b_fragments[slot][token_mma][0],
                                 b_fragments[slot][token_mma][1]);
                    }
                }

                if (next < kKTiles) {
                    __syncthreads();
                    stage_activation(next);
                    cp_commit();
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
                if (row < valid_rows) {
                    keys[item] =
                        score_id_order_key(reusable.reduction.scores[column][local_row], row);
                }
            }
            if constexpr (kRowsPerProducer > 64) {
                if (reducer_lane < kLinearTopK) { keys[2] = top_keys[column][reducer_lane]; }
            }
            M64WarpSort<kRowsPerProducer>(reusable.reduction.sort[reducer_warp])
                .Sort(keys, ScoreIdOrderGreater{});
#pragma unroll
            for (int item = 0; item < kSortItems; ++item) {
                const int rank = reducer_lane * kSortItems + item;
                if (rank < kLinearTopK) {
                    if constexpr (kRowsPerProducer > 64)
                        top_keys[column][rank] = keys[item];
                    else if (column < live_columns) {
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

    if constexpr (kRowsPerProducer > 64) {
        for (int p = tid; p < live_columns * kLinearTopK; p += kThreads) {
            const int column = p / kLinearTopK;
            const int rank   = p - column * kLinearTopK;
            const std::int64_t destination =
                (static_cast<std::int64_t>(column_begin + column) * producer_groups + blockIdx.x) *
                    kLinearTopK +
                rank;
            partial_keys[destination] = top_keys[column][rank];
        }
    }
}

using Launch = void (*)(const Tensor&, const Weight&, std::int32_t, const LinearTopKWorkspace&,
                        cudaStream_t);

template <int TileColumns, int kBlockK = 128, int kRowsPerProducer = 128>
void launch_tile(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    constexpr int shared_bytes =
        sizeof(M64ReusableStorage<TileColumns, kBlockK, kRowsPerProducer>) +
        (kRowsPerProducer > 64 ? TileColumns * kLinearTopK * sizeof(std::uint64_t) : 0);
    if constexpr (shared_bytes > 48 * 1024) {
        CUDA_CHECK(
            cudaFuncSetAttribute(w8_m64_linear_topk_kernel<TileColumns, kBlockK, kRowsPerProducer>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
    }
    w8_m64_linear_topk_kernel<TileColumns, kBlockK, kRowsPerProducer>
        <<<dim3(workspace.producer_groups, div_up(hidden.ne[1], TileColumns)),
           M64Schedule<TileColumns, kBlockK, kRowsPerProducer>::kThreads, shared_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(hidden.data),
            static_cast<const std::uint8_t*>(head.qdata),
            static_cast<const std::uint8_t*>(head.scales), valid_rows,
            static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups,
            hidden.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void linear_topk_w8_m64_launch(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    switch (workspace.tile_columns) {
    case 32:
        return launch_tile<32, 128, 128>(hidden, head, valid_rows, workspace, stream);
    case 40:
        return launch_tile<40, 128, 64>(hidden, head, valid_rows, workspace, stream);
    case 48:
        return launch_tile<48, 128, 64>(hidden, head, valid_rows, workspace, stream);
    case 64:
        return launch_tile<64, 128, 64>(hidden, head, valid_rows, workspace, stream);
    case 80:
        return launch_tile<80, 128, 64>(hidden, head, valid_rows, workspace, stream);
    case 112:
        return launch_tile<112, 64, 64>(hidden, head, valid_rows, workspace, stream);
    case 96:
        return launch_tile<96, 128, 64>(hidden, head, valid_rows, workspace, stream);
    case 120:
        return launch_tile<120, 64, 64>(hidden, head, valid_rows, workspace, stream);
    case 128:
        return launch_tile<128, 64, 64>(hidden, head, valid_rows, workspace, stream);
    }
    throw std::invalid_argument("invalid linear_topk MMA tile");
}
} // namespace ninfer::ops::detail
