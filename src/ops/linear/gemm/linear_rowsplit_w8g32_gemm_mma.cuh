#pragma once

// Dedicated Large-T W8G32 x BF16 Tensor Core GEMM.
//
// out[M,N] = W[M,K] * x[K,N], where W stores one signed int8 code per element
// and one FP16 scale per 32 K elements. Raw codes and four K tiles' worth of
// scales are staged with cp.async before dequantization into a swizzled BF16
// shared tile; x uses a two-stage cp.async pipeline. Tensor Cores execute
// m16n8k16 BF16 MMA with FP32 accumulation.

#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <int BM_, int BN_, int WM_, int WN_, int MIN_BLOCKS_, int STAGES_ = 2>
struct W8G32GemmCfg {
    static constexpr int BM                = BM_;
    static constexpr int BN                = BN_;
    static constexpr int BK                = 64;
    static constexpr int WM                = WM_;
    static constexpr int WN                = WN_;
    static constexpr int MIN_BLOCKS        = MIN_BLOCKS_;
    static constexpr int WARPS_M           = BM / WM;
    static constexpr int WARPS_N           = BN / WN;
    static constexpr int WARPS             = WARPS_M * WARPS_N;
    static constexpr int THREADS           = WARPS * 32;
    static constexpr int MT                = WM / 16;
    static constexpr int NT                = WN / 8;
    static constexpr int KSUB              = BK / 16;
    static constexpr int STAGES            = STAGES_;
    static constexpr int SCALE_CACHE_BYTES = 16;
    static constexpr int SMEM_BYTES =
        BM * BK * 2 + STAGES * BN * BK * 2 + BM * BK + BM * SCALE_CACHE_BYTES;

    static_assert(BM % WM == 0 && BN % WN == 0);
    static_assert(WM % 16 == 0 && WN % 8 == 0);
    static_assert(THREADS <= 1024);
    static_assert(STAGES == 2, "W8G32 MMA uses a two-stage cp.async pipeline");
    static_assert(SMEM_BYTES <= 48 * 1024);
};

__device__ __forceinline__ int w8g32_swz64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <class Cfg, bool FullTiles>
__global__
__launch_bounds__(Cfg::THREADS, Cfg::MIN_BLOCKS) void linear_rowsplit_w8g32_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out, std::int32_t m,
    std::int32_t k, std::int32_t n, std::int32_t padded_k) {
    constexpr int BM   = Cfg::BM;
    constexpr int BN   = Cfg::BN;
    constexpr int BK   = Cfg::BK;
    constexpr int WM   = Cfg::WM;
    constexpr int WN   = Cfg::WN;
    constexpr int MT   = Cfg::MT;
    constexpr int NT   = Cfg::NT;
    constexpr int KSUB = Cfg::KSUB;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[Cfg::STAGES][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[BM * BK];
    __shared__ __align__(16) std::uint8_t Sr[BM * Cfg::SCALE_CACHE_BYTES];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::WARPS_N;
    const int wn   = warp % Cfg::WARPS_N;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int m0 = static_cast<int>(blockIdx.x) * BM;
    const int n0 = static_cast<int>(blockIdx.y) * BN;
    const int kg = padded_k / 32;

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            acc[mi][ni][0] = 0.0f;
            acc[mi][ni][1] = 0.0f;
            acc[mi][ni][2] = 0.0f;
            acc[mi][ni][3] = 0.0f;
        }
    }

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    auto stage_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll 1
        for (int item = tid; item < BN * (BK / 8); item += Cfg::THREADS) {
            const int nl = item / (BK / 8);
            const int k8 = item - nl * (BK / 8);
            const int kk = k0 + k8 * 8;
            const int nn = n0 + nl;
            auto* dst    = &Bs[stage][nl * BK + w8g32_swz64(nl, k8 * 8)];
            if constexpr (FullTiles) {
                cp_async<16, Cache::cg>(dst, &x[static_cast<std::int64_t>(nn) * k + kk]);
            } else {
                const int valid = (nn < n && kk < k) ? min(8, k - kk) * 2 : 0;
                ninfer::ops::cp_async_zfill<16>(
                    dst, &x[static_cast<std::int64_t>(nn < n ? nn : 0) * k + (kk < k ? kk : 0)],
                    valid);
            }
        }
    };

    auto stage_w = [&](int kt) {
        constexpr int GROUPS            = BK / 32;
        constexpr int SCALE_CACHE_TILES = 8 / GROUPS;
        const int g0                    = kt * GROUPS;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += Cfg::THREADS) {
            const int row   = item / (BK / 16);
            const int chunk = item - row * (BK / 16);
            const int grow  = m0 + row;
            auto* dst       = &Cr[row * BK + chunk * 16];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                cp_async<16, Cache::cg>(dst, &codes[gi * 32 + chunk * 16]);
            } else {
                const std::int64_t gi = static_cast<std::int64_t>(grow < m ? grow : 0) * kg + g0;
                ninfer::ops::cp_async_zfill<16>(dst, &codes[gi * 32 + chunk * 16],
                                                grow < m ? 16 : 0);
            }
        }
        if ((kt % SCALE_CACHE_TILES) == 0) {
            for (int row = tid; row < BM; row += Cfg::THREADS) {
                const int grow = m0 + row;
                auto* dst      = &Sr[row * Cfg::SCALE_CACHE_BYTES];
                if constexpr (FullTiles) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                    cp_async<16, Cache::cg>(dst, &scales[gi * 2]);
                } else {
                    const bool valid_row   = grow < m;
                    const int valid_scales = valid_row && g0 < kg ? min(8, kg - g0) : 0;
                    const std::int64_t gi =
                        static_cast<std::int64_t>(valid_row ? grow : 0) * kg + min(g0, kg - 1);
                    ninfer::ops::cp_async_zfill<16>(dst, &scales[gi * 2], valid_scales * 2);
                }
            }
        }
    };

    auto dequant_w = [&](int kt) {
        constexpr int GROUPS            = BK / 32;
        constexpr int SCALE_CACHE_TILES = 8 / GROUPS;
        const int scale_pair_offset     = (kt % SCALE_CACHE_TILES) * GROUPS * 2;
        const int half                  = lane >> 4;
        const int half_lane             = lane & 15;
        for (int row_pair = warp * 2; row_pair < BM; row_pair += Cfg::WARPS * 2) {
            const int row       = row_pair + half;
            unsigned scale_pair = 0;
            if constexpr (GROUPS == 2) {
                scale_pair = half_lane == 0
                                 ? *reinterpret_cast<const std::uint32_t*>(
                                       &Sr[row * Cfg::SCALE_CACHE_BYTES + scale_pair_offset])
                                 : 0;
            } else {
                scale_pair = half_lane == 0
                                 ? *reinterpret_cast<const std::uint16_t*>(
                                       &Sr[row * Cfg::SCALE_CACHE_BYTES + scale_pair_offset])
                                 : 0;
            }
            scale_pair = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
            for (int gg = 0; gg < GROUPS; ++gg) {
                const unsigned scale_bits = (scale_pair >> (gg * 16)) & 0xffffu;
                const float scale         = __half2float(__ushort_as_half(scale_bits));
                const int col             = gg * 32 + half_lane * 2;
                const std::uint16_t packed =
                    *reinterpret_cast<const std::uint16_t*>(&Cr[row * BK + col]);
                const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                const __nv_bfloat162 values = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                                    static_cast<float>(q1) * scale);
                store_vec(&As[row * BK + w8g32_swz64(row, col)], values);
            }
        }
    };

    const int nkt = padded_k / BK;
    stage_x(0, 0);
    stage_w(0);
    ninfer::ops::cp_commit();

#pragma unroll 4
    for (int kt = 0; kt < nkt; ++kt) {
        const int stage = kt % Cfg::STAGES;
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        dequant_w(kt);
        __syncthreads();

        const int next = kt + 1;
        if (next < nkt) {
            stage_x(next % Cfg::STAGES, next);
            stage_w(next);
            ninfer::ops::cp_commit();
        }

        unsigned af[2][MT][4];
        unsigned bf[2][NT][2];
        auto load_fragments = [&](int slot, int ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int ar = wm * WM + mi * 16 + a_rowoff;
                const int ac = ks * 16 + a_coloff;
                ldmatrix_x4(af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                            smem_addr(&As[ar * BK + w8g32_swz64(ar, ac)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int br = wn * WN + ni * 8 + b_rin;
                const int bc = ks * 16 + b_koff;
                ldmatrix_x2(bf[slot][ni][0], bf[slot][ni][1],
                            smem_addr(&Bs[stage][br * BK + w8g32_swz64(br, bc)]));
            }
        };

        load_fragments(0, 0);
#pragma unroll
        for (int ks = 0; ks < KSUB; ++ks) {
            const int slot = ks & 1;
            if (ks + 1 < KSUB) { load_fragments(slot ^ 1, ks + 1); }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3],
                             af[slot][mi][0], af[slot][mi][1], af[slot][mi][2], af[slot][mi][3],
                             bf[slot][ni][0], bf[slot][ni][1]);
                }
            }
        }
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int r0 = m0 + wm * WM + mi * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int c0   = n0 + wn * WN + ni * 8 + 2 * lid;
            const int c1   = c0 + 1;
            const float* a = acc[mi][ni];
            if constexpr (FullTiles) {
                out[static_cast<std::int64_t>(c0) * m + r0] = __float2bfloat16_rn(a[0]);
                out[static_cast<std::int64_t>(c1) * m + r0] = __float2bfloat16_rn(a[1]);
                out[static_cast<std::int64_t>(c0) * m + r1] = __float2bfloat16_rn(a[2]);
                out[static_cast<std::int64_t>(c1) * m + r1] = __float2bfloat16_rn(a[3]);
            } else {
                if (r0 < m && c0 < n) {
                    out[static_cast<std::int64_t>(c0) * m + r0] = __float2bfloat16_rn(a[0]);
                }
                if (r0 < m && c1 < n) {
                    out[static_cast<std::int64_t>(c1) * m + r0] = __float2bfloat16_rn(a[1]);
                }
                if (r1 < m && c0 < n) {
                    out[static_cast<std::int64_t>(c0) * m + r1] = __float2bfloat16_rn(a[2]);
                }
                if (r1 < m && c1 < n) {
                    out[static_cast<std::int64_t>(c1) * m + r1] = __float2bfloat16_rn(a[3]);
                }
            }
        }
    }
}

} // namespace ninfer::ops::detail
