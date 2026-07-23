#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/gdn_input_proj/gdn_conv_snapshot.cuh"
#include "ops/linear/w8/w8_rowsplit_gemm_exact_t_splitk.cuh"
#include "ops/linear/w8/w8_rowsplit_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows                    = 12288;
constexpr int kHidden                  = 2048;
constexpr int kTileK                   = 64;
constexpr int kWarps                   = 8;
constexpr int kMmaRows                 = 16;
constexpr int kRowsPerCta              = 16;
constexpr int kFirstExactCols          = 2;
constexpr int kLastProjectionExactCols = 32;
constexpr int kLastSnapshotExactCols   = 16;
using Output                           = W8SplitOutput2<8192, 4096>;

struct W8GdnSplitKConvEpilogue {
    GdnConvSnapshotEpilogue conv;
    __nv_bfloat16* z;

    template <int ActiveCols>
    __device__ __forceinline__ void store(std::int32_t row,
                                          const float (&projected)[ActiveCols]) const {
        if (row < 8192) {
            conv.store(row, projected);
        } else {
#pragma unroll
            for (int token = 0; token < ActiveCols; ++token) {
                z[static_cast<std::int64_t>(token) * 4096 + row - 8192] =
                    __float2bfloat16_rn(projected[token]);
            }
        }
    }
};

__device__ __forceinline__ int swizzle_64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

union Bf16PairBits {
    __nv_bfloat162 pair;
    unsigned bits;
};

__device__ __forceinline__ unsigned bf16_pair_from_s8(unsigned values) {
    Bf16PairBits biased;
    biased.bits          = __byte_perm(values, 0x43004300u, 0x7150) & 0xff7fff7fu;
    const unsigned signs = (values & 0x80u) | ((values & 0x8000u) << 8);
    Bf16PairBits bias;
    bias.bits = 0x43004300u | signs;
    Bf16PairBits result;
    result.pair = __hsub2_rn(biased.pair, bias.pair);
    return result.bits;
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
__global__
__launch_bounds__(KSplits* NGroups * 32, MinBlocks) void w8_gdn_input_medium_t_splitk_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, int active_cols) {
    constexpr int kKernelWarps = KSplits * NGroups;
    constexpr int kGroupK      = KSplits * kTileK;
    constexpr int kGroups      = kHidden / kGroupK;
    constexpr int kWarpCols    = TileCols / NGroups;
    constexpr int kNt          = kWarpCols / 8;
    constexpr unsigned kMask   = 0xffffffffu;
    static_assert(KSplits == 2 || KSplits == 4);
    static_assert(TileCols % NGroups == 0 && kWarpCols % 8 == 0);
    static_assert(kHidden % kGroupK == 0 && kKernelWarps <= 32);

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
            auto* dst     = &b_shared[warp][col * kTileK + swizzle_64(col, k8 * 8)];
            cp_async<16, Cache::cg>(
                dst, &x[static_cast<std::int64_t>(n_base + col) * kHidden + k0 + k8 * 8]);
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
                                    codes + static_cast<std::int64_t>(cta_row0 + row) * kHidden +
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
                scales + (static_cast<std::int64_t>(scale_row) * (kHidden / 32) + k0 / 32) * 2);
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
                const unsigned af0 = bf16_pair_from_s8(load_code_pair(gid, code_col));
                const unsigned af1 = bf16_pair_from_s8(load_code_pair(gid + 8, code_col));
                const unsigned af2 = bf16_pair_from_s8(load_code_pair(gid, code_col + 8));
                const unsigned af3 = bf16_pair_from_s8(load_code_pair(gid + 8, code_col + 8));
#pragma unroll
                for (int ni = 0; ni < kNt; ++ni) {
                    unsigned bf0, bf1;
                    const int br = ni * 8 + b_rin;
                    ldmatrix_x2(
                        bf0, bf1,
                        smem_addr(&b_shared[warp][br * kTileK + swizzle_64(br, ks * 16 + b_koff)]));
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
            if constexpr (KSplits == 4) {
                if (k_split == 2) {
                    store_vec(partial + ((warp * kNt + ni) * 32 + lane) * 4,
                              make_float4(acc[ni][0], acc[ni][1], acc[ni][2], acc[ni][3]));
                }
            }
        }
    }

    if constexpr (KSplits == 4) {
        __syncthreads();
        if (k_split == 0) {
#pragma unroll
            for (int ni = 0; ni < kNt; ++ni) {
                const int partner_warp = n_group * KSplits + 2;
                const float4 partner =
                    load_vec<float4>(partial + ((partner_warp * kNt + ni) * 32 + lane) * 4);
                acc[ni][0] += partner.x;
                acc[ni][1] += partner.y;
                acc[ni][2] += partner.z;
                acc[ni][3] += partner.w;
            }
        }
    }

    if (k_split == 0) {
        const W8OutputTile output_tile = output.tile(cta_row0);
#pragma unroll
        for (int ni = 0; ni < kNt; ++ni) {
            const int col0 = n_base + ni * 8 + 2 * lid;
            if (col0 < active_cols) {
                *output_tile.at(cta_row0 + gid, col0)     = __float2bfloat16_rn(acc[ni][0]);
                *output_tile.at(cta_row0 + gid + 8, col0) = __float2bfloat16_rn(acc[ni][2]);
            }
            if (col0 + 1 < active_cols) {
                *output_tile.at(cta_row0 + gid, col0 + 1)     = __float2bfloat16_rn(acc[ni][1]);
                *output_tile.at(cta_row0 + gid + 8, col0 + 1) = __float2bfloat16_rn(acc[ni][3]);
            }
        }
    }
}

template <int ActiveCols>
void launch_active_cols(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                        cudaStream_t stream) {
    constexpr int TileCols =
        ActiveCols <= 8 ? 8 : (ActiveCols <= 16 ? 16 : (ActiveCols <= 24 ? 24 : 32));
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    w8_rowsplit_exact_t_splitk_kernel<kHidden, TileCols, ActiveCols>
        <<<kRows / kRowsPerCta, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
}

template <int ActiveCols>
void launch_active_cols_conv_snapshot(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& initial_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream) {
    static_assert(ActiveCols >= 2 && ActiveCols <= 16);
    constexpr int TileCols = ActiveCols <= 8 ? 8 : 16;
    const Output ignored_output{static_cast<__nv_bfloat16*>(query.data),
                                static_cast<__nv_bfloat16*>(z.data)};
    const W8GdnSplitKConvEpilogue epilogue{
        {
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            static_cast<__nv_bfloat16*>(query.data),
            static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data),
            8192,
            2048,
            2048,
            4096,
            0,
        },
        static_cast<__nv_bfloat16*>(z.data),
    };
    w8_rowsplit_exact_t_splitk_kernel<kHidden, TileCols, ActiveCols, Output,
                                      W8GdnSplitKConvEpilogue>
        <<<kRows / kRowsPerCta, kWarps * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), ignored_output, epilogue);
}

template <int TileCols, int KSplits, int NGroups, int MinBlocks>
void launch_medium_cols(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                        cudaStream_t stream) {
    static_assert((8192 % kRowsPerCta) == 0 && (4096 % kRowsPerCta) == 0);
    const Output output{static_cast<__nv_bfloat16*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)};
    w8_gdn_input_medium_t_splitk_kernel<TileCols, KSplits, NGroups, MinBlocks>
        <<<kRows / kRowsPerCta, KSplits * NGroups * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output, x.ne[1]);
}

using ProjectionLauncher = void (*)(const Tensor&, const Weight&, Tensor&, Tensor&, cudaStream_t);
using SnapshotLauncher   = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&,
                                  const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&, cudaStream_t);

template <std::size_t... Offsets>
constexpr auto make_projection_launchers(std::index_sequence<Offsets...>) {
    return std::array<ProjectionLauncher, sizeof...(Offsets)>{
        &launch_active_cols<kFirstExactCols + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_snapshot_launchers(std::index_sequence<Offsets...>) {
    return std::array<SnapshotLauncher, sizeof...(Offsets)>{
        &launch_active_cols_conv_snapshot<kFirstExactCols + static_cast<int>(Offsets)>...};
}

constexpr auto kProjectionLaunchers = make_projection_launchers(
    std::make_index_sequence<kLastProjectionExactCols - kFirstExactCols + 1>{});
constexpr auto kSnapshotLaunchers = make_snapshot_launchers(
    std::make_index_sequence<kLastSnapshotExactCols - kFirstExactCols + 1>{});

} // namespace

void w8_gdn_input_splitk_mma_launch(W8KernelVariant variant, const Tensor& x, const Weight& weight,
                                    Tensor& qkv, Tensor& z, cudaStream_t stream) {
    if (variant != W8KernelVariant::None) {
        throw std::invalid_argument("W8 GDN split-K MMA requires the exact direct variant");
    }
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > 96) {
        throw std::invalid_argument("W8 GDN split-K MMA requires T=2..96");
    }
    if (cols <= kLastProjectionExactCols) {
        kProjectionLaunchers[cols - kFirstExactCols](x, weight, qkv, z, stream);
    } else if (cols <= 48) {
        launch_medium_cols<48, 4, 2, 3>(x, weight, qkv, z, stream);
    } else if (cols <= 64) {
        launch_medium_cols<64, 4, 2, 2>(x, weight, qkv, z, stream);
    } else {
        launch_medium_cols<96, 2, 4, 3>(x, weight, qkv, z, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

void w8_gdn_input_splitk_conv_snapshot_launch(const Tensor& x, const Weight& weight,
                                              const Tensor& conv_weight, Tensor& conv_states,
                                              const Tensor& initial_slot, Tensor& query,
                                              Tensor& key, Tensor& value, Tensor& z,
                                              cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    if (cols < kFirstExactCols || cols > kLastSnapshotExactCols) {
        throw std::invalid_argument("W8 fused GDN input snapshot requires T=2..16");
    }
    kSnapshotLaunchers[cols - kFirstExactCols](x, weight, conv_weight, conv_states, initial_slot,
                                               query, key, value, z, stream);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
