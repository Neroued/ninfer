#pragma once

// Q8G32 RowSplit K-split MMA contraction.
//
// Geometry and scheduling are compile-time values; columns may be exact or capacity-bounded.
// K-split warps
// cooperatively own one 16-row output tile; each warp evaluates a disjoint 64-wide K slice, then
// the CTA reduces FP32 partials in shared memory. Output owns physical row/token addressing; an
// optional caller epilogue may instead consume the FP32 tile.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_schedule.cuh"
#include "ops/linear/q8/q8_operands.h"
#include "ops/linear/q8/q8_shared.cuh"
#include "ops/linear/common/epilogue.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct Q8SlicedKIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

__device__ __forceinline__ int q8_sliced_k_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Q8Bf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned q8_bf16_pair_from_s8(unsigned values) {
    Q8Bf16PairBits biased;
    biased.bits          = __byte_perm(values, 0x43004300u, 0x7150) & 0xff7fff7fu;
    const unsigned signs = (values & 0x80u) | ((values & 0x8000u) << 8);
    Q8Bf16PairBits bias;
    bias.bits = 0x43004300u | signs;
    Q8Bf16PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

// The contraction owns the shared layout; tiled launchers use the same type for opt-in capacity.
template <class Schedule>
union alignas(16) Q8SlicedKSharedStorage {
    struct {
        std::uint8_t codes[Schedule::kBlockRows][Schedule::kBlockK];
        __nv_bfloat16 activations[Schedule::kKWarps][Schedule::kBlockTokens * Schedule::kWarpK];
        std::uint8_t scales[Schedule::kBlockRows][Schedule::kScaleAccess == Q8ScaleAccess::Shared
                                                      ? Schedule::kScaleBytesPerRow
                                                      : 1];
    } staging[Schedule::kStages];

    float partial[Schedule::kKWarps * (Schedule::kBlockTokens / 8) * 32 * 4];
};

struct Q8IdentityColumns {
    __device__ __forceinline__ int operator()(int column) const { return column; }
};

template <class Schedule, bool FullWeights, class Output, class Epilogue,
          class RowPolicy = Q8SlicedKIdentityRows, class ColumnPolicy = Q8IdentityColumns>
__device__ __forceinline__ void q8_a16_sliced_k_mma(Q8LinearOperands operands, Output output,
                                                    Epilogue epilogue, RowPolicy row_policy = {},
                                                    int token_begin            = 0,
                                                    ColumnPolicy column_policy = {}) {
    constexpr int ActiveCols        = Schedule::kTokenCapacity;
    const auto* __restrict__ x      = operands.x;
    const auto* __restrict__ codes  = operands.codes;
    const auto* __restrict__ scales = operands.scales;
    const int columns               = Schedule::kExactTokens ? ActiveCols : operands.tokens;
    const int column_offset =
        Schedule::kExactTokens ? 0 : token_begin + static_cast<int>(blockIdx.y) * ActiveCols;
    const int live_columns =
        Schedule::kExactTokens ? ActiveCols : min(ActiveCols, columns - column_offset);
    const int kHidden             = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.k;
    const int padded_k            = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.padded_k;
    constexpr int kTileK          = Schedule::kWarpK;
    constexpr int kWarps          = Schedule::kKWarps;
    constexpr int kMmaRows        = Schedule::kBlockRows;
    constexpr int kRowsPerCta     = Schedule::kBlockRows;
    constexpr int kGroupK         = Schedule::kBlockK;
    const int kGroups             = (kHidden + kGroupK - 1) / kGroupK;
    constexpr int kTileCols       = Schedule::kBlockTokens;
    constexpr bool kRuntimeActive = Schedule::kActivationStage == Q8ActivationStage::RuntimeActive;
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);
    constexpr int kNt        = kTileCols / 8;
    constexpr unsigned kMask = 0xffffffffu;

    using SharedStorage = Q8SlicedKSharedStorage<Schedule>;

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    auto& shared   = *reinterpret_cast<SharedStorage*>(q8_shared_storage<sizeof(SharedStorage)>());
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int k_split = warp;

    const int cta_row0 = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;

    const auto stage_x = [&](int slot, int group_k0) {
        auto& b_shared              = shared.staging[slot].activations;
        constexpr bool kPaddedStage = Schedule::kActivationStage == Q8ActivationStage::PaddedZero;
        constexpr int kStageCols    = kPaddedStage ? kTileCols : ActiveCols;
        const int stage_columns     = kRuntimeActive ? live_columns : kStageCols;
        const int items_per_split   = stage_columns * (kTileK / 8);
        for (int item = lane; item < items_per_split; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + q8_sliced_k_swizzle_64(col, k8 * 8)];
            const int kk  = group_k0 + warp * kTileK + k8 * 8;
            if constexpr (FullWeights) {
                if constexpr (!Schedule::kExactTokens && !kRuntimeActive) {
                    const int source_col = col < live_columns ? col : 0;
                    cp_async_zfill<16, Schedule::kActivationCache>(
                        dst,
                        &x[static_cast<std::int64_t>(column_policy(column_offset + source_col)) *
                               kHidden +
                           kk],
                        col < live_columns ? 16 : 0);
                } else if constexpr (kRuntimeActive || !kPaddedStage || ActiveCols == kTileCols) {
                    cp_async<16, Schedule::kActivationCache>(
                        dst,
                        &x[static_cast<std::int64_t>(column_policy(column_offset + col)) * kHidden +
                           kk]);
                } else {
                    const int source_col = col < ActiveCols ? col : 0;
                    cp_async_zfill<16, Schedule::kActivationCache>(
                        dst,
                        &x[static_cast<std::int64_t>(column_policy(column_offset + source_col)) *
                               kHidden +
                           kk],
                        col < ActiveCols ? 16 : 0);
                }
            } else {
                const bool valid     = col < live_columns && kk < kHidden;
                const int source_col = valid ? col : 0;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst,
                    &x[static_cast<std::int64_t>(column_policy(column_offset + source_col)) *
                           kHidden +
                       (valid ? kk : 0)],
                    valid ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int slot, int group_k0) {
        auto& code_shared  = shared.staging[slot].codes;
        auto& scale_shared = shared.staging[slot].scales;
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(cta_row0, row);
            for (int chunk = lane; chunk < kGroupK / 16; chunk += 32) {
                const int swizzled_chunk = chunk ^ (row & 7);
                auto* dst                = &code_shared[row][swizzled_chunk * 16];
                if (FullWeights ||
                    (weight_row < operands.rows && group_k0 + chunk * 16 < padded_k)) {
                    cp_async<16, Schedule::kWeightCache>(
                        dst, codes + static_cast<std::int64_t>(weight_row) * padded_k + group_k0 +
                                 chunk * 16);
                } else
                    store_vec(dst, make_uint4(0, 0, 0, 0));
            }
        }
        if constexpr (Schedule::kScaleAccess == Q8ScaleAccess::Shared) {
            constexpr int chunk_bytes = Schedule::kScaleBytesPerRow >= 16 ? 16 : 8;
            constexpr int chunks      = Schedule::kScaleBytesPerRow / chunk_bytes;
            for (int item = tid; item < kMmaRows * chunks; item += kWarps * 32) {
                const int row = item / chunks, chunk = item % chunks;
                const int weight_row = row_policy.weight_row(cta_row0, row);
                auto* dst            = &scale_shared[row][chunk * chunk_bytes];
                const int group      = group_k0 / 32 + chunk * (chunk_bytes / 2);
                if (FullWeights ||
                    (weight_row < operands.rows && group + chunk_bytes / 2 <= padded_k / 32 &&
                     (chunk_bytes == 8 || padded_k % 256 == 0))) {
                    const auto* src = scales +
                                      static_cast<std::int64_t>(weight_row) * (padded_k / 16) +
                                      group * 2;
                    if constexpr (chunk_bytes == 16)
                        cp_async<16, Schedule::kWeightCache>(dst, src);
                    else
                        cp_async<8, Cache::ca>(dst, src);
                } else {
#pragma unroll
                    for (int pair = 0; pair < chunk_bytes / 4; ++pair) {
                        unsigned value = 0;
                        if (weight_row < operands.rows && group + pair * 2 < padded_k / 32)
                            value = load_ldg<unsigned>(
                                scales + static_cast<std::int64_t>(weight_row) * (padded_k / 16) +
                                group * 2 + pair * 4);
                        *reinterpret_cast<unsigned*>(dst + pair * 4) = value;
                    }
                }
            }
        }
    };

    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_koff = k_split * kTileK;
    float acc[kNt][4];
#pragma unroll
    for (int ni = 0; ni < kNt; ++ni) {
        acc[ni][0] = 0.0f;
        acc[ni][1] = 0.0f;
        acc[ni][2] = 0.0f;
        acc[ni][3] = 0.0f;
    }

    stage_codes(0, 0);
    stage_x(0, 0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    constexpr int kGroupUnroll =
        Schedule::kStaticK > 0 && Schedule::kStaticK <= 6144 ? Schedule::kStaticK / kGroupK : 12;
#pragma unroll kGroupUnroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
        const int slot     = group_index % Schedule::kStages;
        auto& code_shared  = shared.staging[slot].codes;
        auto& b_shared     = shared.staging[slot].activations;
        auto& scale_shared = shared.staging[slot].scales;
        if constexpr (Schedule::kStages == 2) {
            if (group_index + 1 < kGroups) {
                stage_codes(slot ^ 1, group_k0 + kGroupK);
                stage_x(slot ^ 1, group_k0 + kGroupK);
                cp_commit();
            }
        }

        unsigned lane_scale_pair = 0;
        if (lid < 2) {
            if constexpr (Schedule::kScaleAccess == Q8ScaleAccess::Shared) {
                const int scale_row = gid + lid * 8;
                lane_scale_pair =
                    *reinterpret_cast<const unsigned*>(&scale_shared[scale_row][warp_koff / 16]);
            } else {
                const int scale_row = row_policy.weight_row(cta_row0, gid + lid * 8);
                if (FullWeights || (scale_row < operands.rows && group_k0 + warp_koff < padded_k))
                    lane_scale_pair = load_ldg<unsigned>(
                        scales + static_cast<std::int64_t>(scale_row) * (padded_k / 16) +
                        (group_k0 + warp_koff) / 16);
            }
        }
        const unsigned top_scale_pair = __shfl_sync(kMask, lane_scale_pair, lane & ~3);
        const unsigned bot_scale_pair = __shfl_sync(kMask, lane_scale_pair, (lane & ~3) + 1);

#pragma unroll
        for (int group = 0; group < 2; ++group) {
            float group_acc[kNt][4];
#pragma unroll
            for (int ni = 0; ni < kNt; ++ni) {
                group_acc[ni][0] = 0.0f;
                group_acc[ni][1] = 0.0f;
                group_acc[ni][2] = 0.0f;
                group_acc[ni][3] = 0.0f;
            }
#pragma unroll
            for (int ki = 0; ki < 2; ++ki) {
                const int ks              = group * 2 + ki;
                const int code_col        = ks * 16 + lid * 2;
                const auto load_code_pair = [&](int code_row, int col) {
                    const int chunk  = (warp_koff + col) >> 4;
                    const int offset = (chunk ^ (code_row & 7)) * 16 + (col & 15);
                    return static_cast<unsigned>(
                        *reinterpret_cast<const unsigned short*>(&code_shared[code_row][offset]));
                };
                const unsigned af0 = q8_bf16_pair_from_s8(load_code_pair(gid, code_col));
                const unsigned af1 = q8_bf16_pair_from_s8(load_code_pair(gid + 8, code_col));
                const unsigned af2 = q8_bf16_pair_from_s8(load_code_pair(gid, code_col + 8));
                const unsigned af3 = q8_bf16_pair_from_s8(load_code_pair(gid + 8, code_col + 8));
#pragma unroll
                for (int ni = 0; ni < kNt; ++ni) {
                    unsigned bf0, bf1;
                    const int br = ni * 8 + b_rin;
                    ldmatrix_x2(
                        bf0, bf1,
                        smem_addr(&b_shared[k_split][br * kTileK + q8_sliced_k_swizzle_64(
                                                                       br, ks * 16 + b_koff)]));
                    mma_bf16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2], group_acc[ni][3],
                             af0, af1, af2, af3, bf0, bf1);
                }
            }
            const unsigned top_bits = group == 0 ? top_scale_pair & 0xffffu : top_scale_pair >> 16;
            const unsigned bot_bits = group == 0 ? bot_scale_pair & 0xffffu : bot_scale_pair >> 16;
            const float top_scale   = (FullWeights || group_k0 + warp_koff + group * 32 < kHidden)
                                          ? __half2float(__ushort_as_half(top_bits))
                                          : 0.0f;
            const float bot_scale   = (FullWeights || group_k0 + warp_koff + group * 32 < kHidden)
                                          ? __half2float(__ushort_as_half(bot_bits))
                                          : 0.0f;
#pragma unroll
            for (int ni = 0; ni < kNt; ++ni) {
                acc[ni][0] = fmaf(group_acc[ni][0], top_scale, acc[ni][0]);
                acc[ni][1] = fmaf(group_acc[ni][1], top_scale, acc[ni][1]);
                acc[ni][2] = fmaf(group_acc[ni][2], bot_scale, acc[ni][2]);
                acc[ni][3] = fmaf(group_acc[ni][3], bot_scale, acc[ni][3]);
            }
        }

        if (group_index + 1 < kGroups) {
            __syncthreads();
            if constexpr (Schedule::kStages == 1) {
                stage_codes(0, group_k0 + kGroupK);
                stage_x(0, group_k0 + kGroupK);
                cp_commit();
            }
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = shared.partial;
    if ((k_split & 1) != 0) {
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            store_vec(partial + ((warp * kNt + ni) * 32 + lane) * 4,
                      make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
        }
    }
    __syncthreads();

    if ((k_split & 1) == 0) {
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            const float4 partner =
                load_vec<float4>(partial + (((warp + 1) * kNt + ni) * 32 + lane) * 4);
            acc[ni][0] += partner.x;
            acc[ni][1] += partner.y;
            acc[ni][2] += partner.z;
            acc[ni][3] += partner.w;
            if (k_split != 0) {
                store_vec(partial + ((warp * kNt + ni) * 32 + lane) * 4,
                          make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
            }
        }
    }
    __syncthreads();

    if (k_split == 0) {
        float* projected = partial;
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            float4 sum = make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]);
#pragma unroll
            for (int split = 2; split < kWarps; split += 2) {
                const float4 value =
                    load_vec<float4>(partial + ((split * kNt + ni) * 32 + lane) * 4);
                sum.x += value.x;
                sum.y += value.y;
                sum.z += value.z;
                sum.w += value.w;
            }
            const int col0 = ni * 8 + 2 * lid;
            if constexpr (requires {
                              epilogue.store_fragment(output, cta_row0 + gid, column_offset + col0,
                                                      sum, operands.rows,
                                                      column_offset + live_columns);
                          }) {
                epilogue.store_fragment(output, cta_row0 + gid, column_offset + col0, sum,
                                        operands.rows, column_offset + live_columns);
            } else if constexpr (requires {
                                     epilogue.apply_row(
                                         output, cta_row0, column_offset,
                                         *reinterpret_cast<const float(*)[ActiveCols]>(partial),
                                         live_columns);
                                 }) {
                if (col0 < ActiveCols) {
                    projected[gid * kTileCols + col0]       = sum.x;
                    projected[(gid + 8) * kTileCols + col0] = sum.z;
                }
                if (col0 + 1 < ActiveCols) {
                    projected[gid * kTileCols + col0 + 1]       = sum.y;
                    projected[(gid + 8) * kTileCols + col0 + 1] = sum.w;
                }
            } else {
                const auto tile =
                    linear_output_tile<RowPolicy::kOutputRowsPerCta>(output, cta_row0);
                const auto store = [&](int row, int col, float value) {
                    if ((FullWeights || row < operands.rows) && col < live_columns)
                        tile.store(row, column_offset + col,
                                   epilogue.apply(row, column_offset + col, value));
                };
                store(cta_row0 + gid, col0, sum.x);
                store(cta_row0 + gid + 8, col0, sum.z);
                store(cta_row0 + gid, col0 + 1, sum.y);
                store(cta_row0 + gid + 8, col0 + 1, sum.w);
            }
        }
        if constexpr (requires {
                          epilogue.apply_row(output, cta_row0, column_offset,
                                             *reinterpret_cast<const float(*)[ActiveCols]>(partial),
                                             live_columns);
                      }) {
            __syncwarp();
            if (lane < kRowsPerCta && (FullWeights || cta_row0 + lane < operands.rows)) {
                float row_values[ActiveCols];
#pragma unroll
                for (int token = 0; token < ActiveCols; ++token) {
                    row_values[token] = projected[lane * kTileCols + token];
                }
                linear_finish_row(output, epilogue, cta_row0 + lane, column_offset, row_values,
                                  live_columns);
            }
        }
    }
}

// Multi-layer fused Ops call the device contraction after selecting their weight view.
template <class Schedule, bool FullWeights, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q8_a16_sliced_k_mma_kernel(
    Q8LinearOperands operands, Output output, Epilogue epilogue, RowPolicy row_policy,
    int token_begin) {
    q8_a16_sliced_k_mma<Schedule, FullWeights>(operands, output, epilogue, row_policy, token_begin);
}
} // namespace ninfer::ops::detail
