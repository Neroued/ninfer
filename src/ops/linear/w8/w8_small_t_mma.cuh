#pragma once

// Reusable W8G32 RowSplit exact-small-T MMA core.
//
// Geometry, active tokens, and scheduling are independent compile-time values. K-split warps
// cooperatively own one 16-row output tile; each warp evaluates a disjoint 64-wide K slice, then
// the CTA reduces FP32 partials in shared memory. Output owns physical row/token addressing; an
// optional caller epilogue may instead consume the FP32 tile.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_rowsplit_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct W8SmallTMmaStoreEpilogue {};

struct W8SmallTMmaResidualEpilogue {};

struct W8SmallTMmaIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

__device__ __forceinline__ int w8_small_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union W8SmallTBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned w8_small_t_bf16_pair_from_s8(unsigned values) {
    W8SmallTBf16PairBits biased;
    biased.bits          = __byte_perm(values, 0x43004300u, 0x7150) & 0xff7fff7fu;
    const unsigned signs = (values & 0x80u) | ((values & 0x8000u) << 8);
    W8SmallTBf16PairBits bias;
    bias.bits = 0x43004300u | signs;
    W8SmallTBf16PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

template <class Geometry, int ActiveCols, class Schedule, class Output,
          class Epilogue = W8SmallTMmaStoreEpilogue, class RowPolicy = W8SmallTMmaIdentityRows,
          bool DirectPairEpilogue = false>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void w8_small_t_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, Epilogue epilogue = {},
    RowPolicy row_policy = {}) {
// ldmatrix (sm_75+) + mma.m16n8k16 (sm_80+), same as every other tensor-core kernel this
// port has hit. Unlike rowsplit_grouped_mma.cuh / bf16_gdn_gating_proj_gemm_mma.cuh, this
// one genuinely IS on the decode-critical path (vocabulary/gate-up/down/attention-in-out
// projections — see docs/v100.md), so it doesn't just get trapped and forgotten:
// launch_w8_small_t (w8_small_t.cu) routes to the already-Volta-validated warp-per-row
// SIMT kernel (w8_rowsplit_gemm_simt.cuh) instead below sm_80, via the NINFER_VOLTA_BUILD
// host-side signal (CMakeLists.txt) — __CUDA_ARCH__ isn't visible in that host function,
// so the kernel body itself still needs its own guard purely so it compiles.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr int kHidden     = Geometry::kInputRows;
    constexpr int kTileK      = Schedule::kTileKPerWarp;
    constexpr int kWarps      = Schedule::kKWarps;
    constexpr int kMmaRows    = Schedule::kRowsPerCta;
    constexpr int kRowsPerCta = Schedule::kRowsPerCta;
    constexpr int kGroupK     = Schedule::kGroupK;
    constexpr int kGroups     = kHidden / kGroupK;
    constexpr int kTileCols   = Schedule::kTileTokens;
    static_assert((kHidden % kGroupK) == 0);
    static_assert(ActiveCols >= 1 && ActiveCols <= kTileCols);
    static_assert(RowPolicy::kOutputRowsPerCta <= kRowsPerCta);
    constexpr int kNt        = kTileCols / 8;
    constexpr unsigned kMask = 0xffffffffu;

    union SharedStorage {
        struct {
            std::uint8_t codes[kMmaRows][kGroupK];
            __nv_bfloat16 activations[kWarps][kTileCols * kTileK];
            std::uint8_t scales[kMmaRows][Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared
                                              ? Schedule::kScaleBytesPerRow
                                              : 1];
        } staging;

        float partial[kWarps * kNt * 32 * 4];
    };

    __shared__ __align__(16) SharedStorage shared;
    auto& code_shared  = shared.staging.codes;
    auto& b_shared     = shared.staging.activations;
    auto& scale_shared = shared.staging.scales;

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = lane >> 2;
    const int lid     = lane & 3;
    const int k_split = warp;

    const int cta_row0 = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;

    const auto stage_x = [&](int group_k0) {
        constexpr bool kPaddedStage =
            Schedule::kActivationStage == W8SmallTMmaActivationStage::PaddedZero;
        constexpr int kStageCols     = kPaddedStage ? kTileCols : ActiveCols;
        constexpr int kItemsPerSplit = kStageCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + w8_small_t_swizzle_64(col, k8 * 8)];
            if constexpr (!kPaddedStage || ActiveCols == kTileCols) {
                cp_async<16, Schedule::kActivationCache>(
                    dst, &x[static_cast<std::int64_t>(col) * kHidden + group_k0 + warp * kTileK +
                            k8 * 8]);
            } else {
                const int source_col = col < ActiveCols ? col : 0;
                cp_async_zfill<16, Schedule::kActivationCache>(
                    dst,
                    &x[static_cast<std::int64_t>(source_col) * kHidden + group_k0 + warp * kTileK +
                       k8 * 8],
                    col < ActiveCols ? 16 : 0);
            }
        }
    };

    const auto stage_codes = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < Schedule::kRowsPerLoaderWarp; ++row_item) {
            const int row        = warp * Schedule::kRowsPerLoaderWarp + row_item;
            const int weight_row = row_policy.weight_row(cta_row0, row);
            for (int chunk = lane; chunk < kGroupK / 16; chunk += 32) {
                const int swizzled_chunk = chunk ^ (row & 7);
                cp_async<16, Schedule::kWeightCache>(
                    &code_shared[row][swizzled_chunk * 16],
                    codes + static_cast<std::int64_t>(weight_row) * kHidden + group_k0 +
                        chunk * 16);
            }
        }
        if constexpr (Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared) {
            constexpr int kScaleChunksPerRow = Schedule::kScaleBytesPerRow / 16;
            for (int item = tid; item < kMmaRows * kScaleChunksPerRow; item += kWarps * 32) {
                const int row        = item / kScaleChunksPerRow;
                const int chunk      = item - row * kScaleChunksPerRow;
                const int weight_row = row_policy.weight_row(cta_row0, row);
                cp_async<16, Schedule::kWeightCache>(
                    &scale_shared[row][chunk * 16],
                    scales + (static_cast<std::int64_t>(weight_row) * Geometry::kGroupsPerRow +
                              group_k0 / 32 + chunk * 8) *
                                 2);
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

    stage_codes(0);
    stage_x(0);
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    constexpr int kGroupUnroll = kHidden <= 6144 ? kGroups : 12;
#pragma unroll kGroupUnroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;

        unsigned lane_scale_pair = 0;
        if (lid < 2) {
            if constexpr (Schedule::kScaleAccess == W8SmallTMmaScaleAccess::Shared) {
                const int scale_row = gid + lid * 8;
                lane_scale_pair =
                    *reinterpret_cast<const unsigned*>(&scale_shared[scale_row][warp_koff / 16]);
            } else {
                const int scale_row = row_policy.weight_row(cta_row0, gid + lid * 8);
                lane_scale_pair     = *reinterpret_cast<const unsigned*>(
                    scales + (static_cast<std::int64_t>(scale_row) * Geometry::kGroupsPerRow +
                              group_k0 / 32 + warp_koff / 32) *
                                 2);
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
                const unsigned af0 = w8_small_t_bf16_pair_from_s8(load_code_pair(gid, code_col));
                const unsigned af1 =
                    w8_small_t_bf16_pair_from_s8(load_code_pair(gid + 8, code_col));
                const unsigned af2 =
                    w8_small_t_bf16_pair_from_s8(load_code_pair(gid, code_col + 8));
                const unsigned af3 =
                    w8_small_t_bf16_pair_from_s8(load_code_pair(gid + 8, code_col + 8));
#pragma unroll
                for (int ni = 0; ni < kNt; ++ni) {
                    unsigned bf0, bf1;
                    const int br = ni * 8 + b_rin;
                    ldmatrix_x2(
                        bf0, bf1,
                        smem_addr(&b_shared[k_split][br * kTileK +
                                                     w8_small_t_swizzle_64(br, ks * 16 + b_koff)]));
                    mma_bf16(group_acc[ni][0], group_acc[ni][1], group_acc[ni][2], group_acc[ni][3],
                             af0, af1, af2, af3, bf0, bf1);
                }
            }
            const unsigned top_bits = group == 0 ? top_scale_pair & 0xffffu : top_scale_pair >> 16;
            const unsigned bot_bits = group == 0 ? bot_scale_pair & 0xffffu : bot_scale_pair >> 16;
            const float top_scale   = __half2float(__ushort_as_half(top_bits));
            const float bot_scale   = __half2float(__ushort_as_half(bot_bits));
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
            stage_codes(group_k0 + kGroupK);
            stage_x(group_k0 + kGroupK);
            cp_commit();
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
        const W8OutputTile output_tile = output.tile(cta_row0);
        float* projected               = partial;
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
            if constexpr (std::is_same_v<Epilogue, W8SmallTMmaStoreEpilogue> ||
                          std::is_same_v<Epilogue, W8SmallTMmaResidualEpilogue>) {
                const auto store = [&](int row, int col, float value) {
                    __nv_bfloat16* destination = output_tile.at(row, col);
                    if constexpr (std::is_same_v<Epilogue, W8SmallTMmaResidualEpilogue>) {
                        value += __bfloat162float(*destination);
                    }
                    *destination = __float2bfloat16_rn(value);
                };
                if (col0 < ActiveCols) {
                    store(cta_row0 + gid, col0, sum.x);
                    store(cta_row0 + gid + 8, col0, sum.z);
                }
                if (col0 + 1 < ActiveCols) {
                    store(cta_row0 + gid, col0 + 1, sum.y);
                    store(cta_row0 + gid + 8, col0 + 1, sum.w);
                }
            } else if constexpr (DirectPairEpilogue) {
                epilogue.template store_pair<ActiveCols>(cta_row0 + gid, col0, sum);
            } else {
                if (col0 < ActiveCols) {
                    projected[gid * kTileCols + col0]       = sum.x;
                    projected[(gid + 8) * kTileCols + col0] = sum.z;
                }
                if (col0 + 1 < ActiveCols) {
                    projected[gid * kTileCols + col0 + 1]       = sum.y;
                    projected[(gid + 8) * kTileCols + col0 + 1] = sum.w;
                }
            }
        }
        if constexpr (!std::is_same_v<Epilogue, W8SmallTMmaStoreEpilogue> &&
                      !std::is_same_v<Epilogue, W8SmallTMmaResidualEpilogue> &&
                      !DirectPairEpilogue) {
            __syncwarp();
            if (lane < kRowsPerCta) {
                float row_values[ActiveCols];
#pragma unroll
                for (int token = 0; token < ActiveCols; ++token) {
                    row_values[token] = projected[lane * kTileCols + token];
                }
                epilogue.store(cta_row0 + lane, row_values);
            }
        }
    }
#else  // __CUDA_ARCH__ < 800
    (void)x;
    (void)codes;
    (void)scales;
    (void)output;
    (void)epilogue;
    (void)row_policy;
    __trap();
#endif // __CUDA_ARCH__ >= 800
}

} // namespace ninfer::ops::detail
