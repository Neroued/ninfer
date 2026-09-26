#pragma once

// Q8G32 RowSplit medium-T split-K MMA core. K-split warps share one 16-row
// weight tile while N-groups cover disjoint column ranges. Output owns the
// physical direct-write policy.

#include "ops/linear/q8/q8_a16_sliced_k_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <class Schedule, bool FullWeights, class Output, class Epilogue>
__global__ __launch_bounds__(
    Schedule::kThreads,
    Schedule::kMinBlocksPerSm) void q8_a16_grouped_sliced_k_mma_kernel(Q8LinearOperands operands,
                                                                       Output output,
                                                                       Epilogue epilogue,
                                                                       int token_begin,
                                                                       int token_count) {
    constexpr int TileCols = Schedule::kBlockTokens, KSplits = Schedule::kKWarps;
    constexpr int NGroups = Schedule::kTokenGroups, Stages = Schedule::kStages;
    constexpr bool TiledColumns = Schedule::kTiledTokens;
    const int Hidden            = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.k;
    const int padded_k          = Schedule::kStaticK > 0 ? Schedule::kStaticK : operands.padded_k;
    const auto* __restrict__ x  = operands.x;
    const auto* __restrict__ codes  = operands.codes;
    const auto* __restrict__ scales = operands.scales;
    const int active_cols           = token_count;
    constexpr int kTileK            = 64;
    constexpr int kMmaRows          = 16;
    constexpr int kRowsPerCta       = 16;
    constexpr int kKernelWarps      = KSplits * NGroups;
    constexpr int kGroupK           = KSplits * kTileK;
    const int kGroups               = (Hidden + kGroupK - 1) / kGroupK;
    constexpr int kWarpCols         = TileCols / NGroups;
    constexpr int kNt               = kWarpCols / 8;
    constexpr unsigned kMask        = 0xffffffffu;
    static_assert(KSplits == 2 || KSplits == 4 || KSplits == 8);
    static_assert(TileCols % NGroups == 0 && kWarpCols % 8 == 0);
    static_assert(kKernelWarps <= 32);

    struct Shared {
        struct {
            alignas(16) std::uint8_t codes[kMmaRows][kGroupK];
            alignas(16) __nv_bfloat16 activations[kKernelWarps][kWarpCols * kTileK];
        } stage[Stages];
    };

    static_assert(sizeof(Shared) == Schedule::kSharedBytes);
    auto& shared = *reinterpret_cast<Shared*>(q8_shared_storage<sizeof(Shared)>());

    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int n_group     = warp / KSplits;
    const int k_split     = warp - n_group * KSplits;
    const int gid         = lane >> 2;
    const int lid         = lane & 3;
    const int n_base      = n_group * kWarpCols;
    const int token_tiles = TiledColumns ? static_cast<int>(gridDim.y) : 1;
    const int linear_block =
        static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    const int column_offset = TiledColumns ? (linear_block % token_tiles) * TileCols : 0;
    const int tile_columns =
        TiledColumns ? min(TileCols, active_cols - column_offset) : active_cols;
    const int remaining  = tile_columns - n_base;
    const int local_cols = remaining <= 0 ? 0 : (remaining < kWarpCols ? remaining : kWarpCols);
    // Neighboring token tiles share a weight tile in the bulk grid.
    const int cta_row0 =
        (TiledColumns ? linear_block / token_tiles : static_cast<int>(blockIdx.x)) * kRowsPerCta;

    const auto stage_x = [&](int slot, int k0) {
        auto& b_shared = shared.stage[slot].activations;
        for (int item = lane; item < local_cols * (kTileK / 8); item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + q8_sliced_k_swizzle_64(col, k8 * 8)];
            if (FullWeights || k0 + k8 * 8 < Hidden)
                cp_async<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(token_begin + column_offset + n_base + col) *
                                Hidden +
                            k0 + k8 * 8]);
            else
                store_vec(dst, make_uint4(0, 0, 0, 0));
        }
        cp_commit();
    };

    const auto stage_codes = [&](int slot, int group_k0) {
        auto& code_shared     = shared.stage[slot].codes;
        constexpr int kChunks = kGroupK / 16;
        for (int item = tid; item < kMmaRows * kChunks; item += kKernelWarps * 32) {
            const int row            = item / kChunks;
            const int chunk          = item - row * kChunks;
            const int swizzled_chunk = chunk ^ (row & 7);
            auto* dst                = &code_shared[row][swizzled_chunk * 16];
            if (FullWeights || (cta_row0 + row < operands.rows && group_k0 + chunk * 16 < padded_k))
                cp_async<16, Schedule::kWeightCache>(
                    dst, codes + static_cast<std::int64_t>(cta_row0 + row) * padded_k + group_k0 +
                             chunk * 16);
            else
                store_vec(dst, make_uint4(0, 0, 0, 0));
        }
        cp_commit();
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
    stage_x(0, warp_koff);
    cp_wait<0>();
    __syncthreads();

    // Bulk tiles keep the K loop compact; the small fixed callers retain their unrolling.
    constexpr int kGroupUnroll =
        TiledColumns || Schedule::kStaticK == 0 ? 1 : Schedule::kStaticK / kGroupK;
#pragma unroll kGroupUnroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
        const int k0       = group_k0 + warp_koff;
        const int slot     = group_index % Stages;
        auto& code_shared  = shared.stage[slot].codes;
        auto& b_shared     = shared.stage[slot].activations;
        if constexpr (Stages == 2) {
            if (group_index + 1 < kGroups) {
                stage_codes(slot ^ 1, group_k0 + kGroupK);
                stage_x(slot ^ 1, k0 + kGroupK);
            }
        }

        unsigned lane_scale_pair = 0;
        if (lid < 2) {
            const int scale_row = cta_row0 + gid + lid * 8;
            if (FullWeights || (scale_row < operands.rows && k0 < padded_k)) {
                const auto* address =
                    scales + (static_cast<std::int64_t>(scale_row) * (padded_k / 32) + k0 / 32) * 2;
                if constexpr (Schedule::kReadOnlyScales)
                    lane_scale_pair = load_ldg<unsigned>(address);
                else
                    lane_scale_pair = load_vec<unsigned>(address);
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
                        smem_addr(&b_shared[warp][br * kTileK +
                                                  q8_sliced_k_swizzle_64(br, ks * 16 + b_koff)]));
                    mma_bf16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2], group_acc[ni][3],
                             af0, af1, af2, af3, bf0, bf1);
                }
            }
            const unsigned top_bits = group == 0 ? top_scale_pair & 0xffffu : top_scale_pair >> 16;
            const unsigned bot_bits = group == 0 ? bot_scale_pair & 0xffffu : bot_scale_pair >> 16;
            const float top_scale   = (FullWeights || k0 + group * 32 < Hidden)
                                          ? __half2float(__ushort_as_half(top_bits))
                                          : 0.0f;
            const float bot_scale   = (FullWeights || k0 + group * 32 < Hidden)
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
            if constexpr (Stages == 1) {
                stage_codes(0, group_k0 + kGroupK);
                stage_x(0, k0 + kGroupK);
            }
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = reinterpret_cast<float*>(shared.stage[0].activations);
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

    if constexpr (KSplits > 2) {
        __syncthreads();
        if (k_split == 0) {
#pragma unroll
            for (int ni = 0; ni < kNt; ++ni) {
#pragma unroll
                for (int split = 2; split < KSplits; split += 2) {
                    const int partner_warp = n_group * KSplits + split;
                    const float4 partner =
                        load_vec<float4>(partial + ((partner_warp * kNt + ni) * 32 + lane) * 4);
                    acc[ni][0] += partner.x;
                    acc[ni][1] += partner.y;
                    acc[ni][2] += partner.z;
                    acc[ni][3] += partner.w;
                }
            }
        }
    }

    if (k_split == 0) {
        const auto output_tile = linear_output_tile<16>(output, cta_row0);
        const auto store       = [&](int row, int col, float value) {
            const int token = token_begin + column_offset + col;
            if (FullWeights || row < operands.rows)
                output_tile.store(row, token, epilogue.apply(row, token, value));
        };
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            const int col0 = n_base + ni * 8 + 2 * lid;
            if (col0 < tile_columns) {
                store(cta_row0 + gid, col0, acc[ni][0]);
                store(cta_row0 + gid + 8, col0, acc[ni][2]);
            }
            if (col0 + 1 < tile_columns) {
                store(cta_row0 + gid, col0 + 1, acc[ni][1]);
                store(cta_row0 + gid + 8, col0 + 1, acc[ni][3]);
            }
        }
    }
}

} // namespace ninfer::ops::detail
