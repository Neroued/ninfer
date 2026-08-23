#pragma once

// W8G32 RowSplit medium-T split-K MMA core. K-split warps share one 16-row
// weight tile while N-groups cover disjoint column ranges. Output owns the
// physical direct-write policy.

#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int Hidden, int TileCols, int KSplits, int NGroups, int MinBlocks, class Output,
          bool AddResidual = false>
__global__
__launch_bounds__(KSplits* NGroups * 32, MinBlocks) void w8_rowsplit_medium_t_splitk_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, int active_cols) {
// mma_bf16/ldmatrix_x2 (mma.cuh) already trap below sm_80, but that alone isn't enough
// here: this kernel's __shared__ declarations below are large enough (up to ~90KB across
// the TileCols/KSplits/NGroups instantiations this port's callers use) that nvlink
// rejects them outright against Volta's 48KB static-shared-memory cap -- a link-time
// failure, not a ptxas one, and it happens regardless of whether the mma-using code path
// is ever reached, because __shared__ is a static allocation independent of control flow.
// The whole body (declarations included) needs to be behind the guard, not just the mma
// calls within it, unlike every other kernel this port has trap-stubbed so far.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    constexpr int kTileK       = 64;
    constexpr int kMmaRows     = 16;
    constexpr int kRowsPerCta  = 16;
    constexpr int kKernelWarps = KSplits * NGroups;
    constexpr int kGroupK      = KSplits * kTileK;
    constexpr int kGroups      = Hidden / kGroupK;
    constexpr int kWarpCols    = TileCols / NGroups;
    constexpr int kNt          = kWarpCols / 8;
    constexpr unsigned kMask   = 0xffffffffu;
    static_assert(KSplits == 2 || KSplits == 4 || KSplits == 8);
    static_assert(TileCols % NGroups == 0 && kWarpCols % 8 == 0);
    static_assert(Hidden % kGroupK == 0 && kKernelWarps <= 32);

    __shared__ __align__(16) std::uint8_t code_shared[kMmaRows][kGroupK];
    __shared__ __align__(16) __nv_bfloat16 b_shared[kKernelWarps][kWarpCols * kTileK];

    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int n_group    = warp / KSplits;
    const int k_split    = warp - n_group * KSplits;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int n_base     = n_group * kWarpCols;
    const int remaining  = active_cols - n_base;
    const int local_cols = remaining <= 0 ? 0 : (remaining < kWarpCols ? remaining : kWarpCols);
    const int cta_row0   = static_cast<int>(blockIdx.x) * kRowsPerCta;

    const auto stage_x = [&](int k0) {
        for (int item = lane; item < local_cols * (kTileK / 8); item += 32) {
            const int col = item / (kTileK / 8);
            const int k8  = item - col * (kTileK / 8);
            auto* dst     = &b_shared[warp][col * kTileK + w8_small_t_swizzle_64(col, k8 * 8)];
            cp_async<16, Cache::cg>(
                dst, &x[static_cast<std::int64_t>(n_base + col) * Hidden + k0 + k8 * 8]);
        }
        cp_commit();
    };

    const auto stage_codes = [&](int group_k0) {
        constexpr int kChunks = kGroupK / 16;
        for (int item = tid; item < kMmaRows * kChunks; item += kKernelWarps * 32) {
            const int row            = item / kChunks;
            const int chunk          = item - row * kChunks;
            const int swizzled_chunk = chunk ^ (row & 7);
            cp_async<16, Cache::cg>(&code_shared[row][swizzled_chunk * 16],
                                    codes + static_cast<std::int64_t>(cta_row0 + row) * Hidden +
                                        group_k0 + chunk * 16);
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
    stage_x(warp_koff);
    cp_wait<0>();
    __syncthreads();

#pragma unroll
    for (int group_index = 0; group_index < kGroups; ++group_index) {
        const int group_k0 = group_index * kGroupK;
        const int k0       = group_k0 + warp_koff;

        unsigned lane_scale_pair = 0;
        if (lid < 2) {
            const int scale_row = cta_row0 + gid + lid * 8;
            lane_scale_pair     = *reinterpret_cast<const unsigned*>(
                scales + (static_cast<std::int64_t>(scale_row) * (Hidden / 32) + k0 / 32) * 2);
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
                        smem_addr(&b_shared[warp][br * kTileK +
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
            stage_x(k0 + kGroupK);
            cp_wait<0>();
            __syncthreads();
        }
    }

    __syncthreads();
    auto* partial = reinterpret_cast<float*>(b_shared);
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
        const W8OutputTile output_tile = output.tile(cta_row0);
        const auto store               = [&](int row, int col, float value) {
            __nv_bfloat16* destination = output_tile.at(row, col);
            if constexpr (AddResidual) { value += __bfloat162float(*destination); }
            *destination = __float2bfloat16_rn(value);
        };
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            const int col0 = n_base + ni * 8 + 2 * lid;
            if (col0 < active_cols) {
                store(cta_row0 + gid, col0, acc[ni][0]);
                store(cta_row0 + gid + 8, col0, acc[ni][2]);
            }
            if (col0 + 1 < active_cols) {
                store(cta_row0 + gid, col0 + 1, acc[ni][1]);
                store(cta_row0 + gid + 8, col0 + 1, acc[ni][3]);
            }
        }
    }
#else  // __CUDA_ARCH__ < 800
    (void)x;
    (void)codes;
    (void)scales;
    (void)output;
    (void)active_cols;
    __trap();
#endif // __CUDA_ARCH__ >= 800
}

} // namespace ninfer::ops::detail
