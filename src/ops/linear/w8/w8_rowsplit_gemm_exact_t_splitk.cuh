#pragma once

// W8G32 RowSplit exact-small-T split-K MMA core.
//
// Eight warps cooperatively own one 16-row output tile. Each warp evaluates a
// disjoint 64-wide K slice, then the CTA reduces the eight FP32 partials in
// shared memory. ActiveCols is compile-time exact, so T=2..32 has neither a
// runtime column loop nor padded output work. Output owns the physical split
// epilogue; an optional caller epilogue may instead consume the FP32 tile.

#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <type_traits>

namespace ninfer::ops::detail {

struct W8ExactTSplitKStoreEpilogue {};

struct W8ExactTIdentityRows {
    static constexpr int kOutputRowsPerCta = 16;

    __device__ __forceinline__ int weight_row(int output_row0, int local_row) const {
        return output_row0 + local_row;
    }
};

__device__ __forceinline__ int w8_exact_t_swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union W8ExactTBf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned w8_exact_t_bf16_pair_from_s8(unsigned values) {
    W8ExactTBf16PairBits biased;
    biased.bits          = __byte_perm(values, 0x43004300u, 0x7150) & 0xff7fff7fu;
    const unsigned signs = (values & 0x80u) | ((values & 0x8000u) << 8);
    W8ExactTBf16PairBits bias;
    bias.bits = 0x43004300u | signs;
    W8ExactTBf16PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

template <int TileCols>
inline constexpr int kW8ExactTMinBlocks =
    TileCols == 8 ? 5 : (TileCols == 16 ? 4 : (TileCols == 24 ? 3 : 2));

template <int Hidden, int TileCols, int ActiveCols, class Output,
          class Epilogue = W8ExactTSplitKStoreEpilogue, class RowPolicy = W8ExactTIdentityRows>
__global__
__launch_bounds__(8 * 32, kW8ExactTMinBlocks<TileCols>) void w8_rowsplit_exact_t_splitk_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, Epilogue epilogue = {},
    RowPolicy row_policy = {}) {
    constexpr int kTileK      = 64;
    constexpr int kWarps      = 8;
    constexpr int kMmaRows    = 16;
    constexpr int kRowsPerCta = 16;
    constexpr int kGroupK     = kWarps * kTileK;
    constexpr int kGroups     = Hidden / kGroupK;
    static_assert(Hidden % kGroupK == 0);
    static_assert(TileCols == 8 || TileCols == 16 || TileCols == 24 || TileCols == 32);
    static_assert(ActiveCols >= 2 && ActiveCols <= TileCols);
    constexpr int kNt        = TileCols / 8;
    constexpr unsigned kMask = 0xffffffffu;

    union SharedStorage {
        struct {
            std::uint8_t codes[kMmaRows][kGroupK];
            __nv_bfloat16 activations[kWarps][TileCols * kTileK];
            std::uint8_t scales[kMmaRows][ActiveCols > 4 ? kGroupK / 16 : 1];
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
        constexpr int kItemsPerSplit = ActiveCols * (kTileK / 8);
        for (int item = lane; item < kItemsPerSplit; item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + w8_exact_t_swizzle_64(col, k8 * 8)];
            cp_async<16, Cache::ca>(
                dst,
                &x[static_cast<std::int64_t>(col) * Hidden + group_k0 + warp * kTileK + k8 * 8]);
        }
        cp_commit();
    };

    const auto stage_codes = [&](int group_k0) {
#pragma unroll
        for (int row_item = 0; row_item < 2; ++row_item) {
            const int row            = warp * 2 + row_item;
            const int chunk          = lane;
            const int swizzled_chunk = chunk ^ (row & 7);
            const int weight_row     = row_policy.weight_row(cta_row0, row);
            cp_async<16, Cache::cg>(&code_shared[row][swizzled_chunk * 16],
                                    codes + static_cast<std::int64_t>(weight_row) * Hidden +
                                        group_k0 + chunk * 16);
        }
        if constexpr (ActiveCols > 4) {
            for (int item = tid; item < kMmaRows * 2; item += kWarps * 32) {
                const int row        = item >> 1;
                const int chunk      = item & 1;
                const int weight_row = row_policy.weight_row(cta_row0, row);
                cp_async<16, Cache::cg>(&scale_shared[row][chunk * 16],
                                        scales +
                                            (static_cast<std::int64_t>(weight_row) * (Hidden / 32) +
                                             group_k0 / 32 + chunk * 8) *
                                                2);
            }
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

    stage_codes(0);
    stage_x(0);
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;

        unsigned lane_scale_pair = 0;
        if (lid < 2) {
            if constexpr (ActiveCols > 4) {
                const int scale_row = gid + lid * 8;
                lane_scale_pair =
                    *reinterpret_cast<const unsigned*>(&scale_shared[scale_row][warp_koff / 16]);
            } else {
                const int scale_row = row_policy.weight_row(cta_row0, gid + lid * 8);
                lane_scale_pair     = *reinterpret_cast<const unsigned*>(
                    scales + (static_cast<std::int64_t>(scale_row) * (Hidden / 32) + group_k0 / 32 +
                              warp_koff / 32) *
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
                const unsigned af0 = w8_exact_t_bf16_pair_from_s8(load_code_pair(gid, code_col));
                const unsigned af1 =
                    w8_exact_t_bf16_pair_from_s8(load_code_pair(gid + 8, code_col));
                const unsigned af2 =
                    w8_exact_t_bf16_pair_from_s8(load_code_pair(gid, code_col + 8));
                const unsigned af3 =
                    w8_exact_t_bf16_pair_from_s8(load_code_pair(gid + 8, code_col + 8));
#pragma unroll
                for (int ni = 0; ni < kNt; ++ni) {
                    unsigned bf0, bf1;
                    const int br = ni * 8 + b_rin;
                    ldmatrix_x2(
                        bf0, bf1,
                        smem_addr(&b_shared[k_split][br * kTileK +
                                                     w8_exact_t_swizzle_64(br, ks * 16 + b_koff)]));
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
            if constexpr (std::is_same_v<Epilogue, W8ExactTSplitKStoreEpilogue>) {
                if (col0 < ActiveCols) {
                    *output_tile.at(cta_row0 + gid, col0)     = __float2bfloat16_rn(sum.x);
                    *output_tile.at(cta_row0 + gid + 8, col0) = __float2bfloat16_rn(sum.z);
                }
                if (col0 + 1 < ActiveCols) {
                    *output_tile.at(cta_row0 + gid, col0 + 1)     = __float2bfloat16_rn(sum.y);
                    *output_tile.at(cta_row0 + gid + 8, col0 + 1) = __float2bfloat16_rn(sum.w);
                }
            } else {
                if (col0 < ActiveCols) {
                    projected[gid * TileCols + col0]       = sum.x;
                    projected[(gid + 8) * TileCols + col0] = sum.z;
                }
                if (col0 + 1 < ActiveCols) {
                    projected[gid * TileCols + col0 + 1]       = sum.y;
                    projected[(gid + 8) * TileCols + col0 + 1] = sum.w;
                }
            }
        }
        if constexpr (!std::is_same_v<Epilogue, W8ExactTSplitKStoreEpilogue>) {
            __syncwarp();
            if (lane < kRowsPerCta) {
                float row_values[ActiveCols];
#pragma unroll
                for (int token = 0; token < ActiveCols; ++token) {
                    row_values[token] = projected[lane * TileCols + token];
                }
                epilogue.store(cta_row0 + lane, row_values);
            }
        }
    }
}

} // namespace ninfer::ops::detail
