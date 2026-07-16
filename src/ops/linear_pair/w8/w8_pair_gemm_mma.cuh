#pragma once

// Paired MTP K/V W8G32 GEMM. One CTA keeps a single BF16 activation tile in
// shared memory and contracts it against independent K and V weight tiles.

#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_launch.h"

namespace ninfer::ops::detail {

template <W8KernelVariant Variant>
__global__ __launch_bounds__(256, 2) void w8_pair_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ k_codes,
    const std::uint8_t* __restrict__ k_scales, const std::uint8_t* __restrict__ v_codes,
    const std::uint8_t* __restrict__ v_scales, __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out, std::int32_t m, std::int32_t k, std::int32_t n,
    std::int32_t padded_k) {
    static_assert(Variant == W8KernelVariant::Full || Variant == W8KernelVariant::Predicated);
    constexpr bool FullTiles        = Variant == W8KernelVariant::Full;
    constexpr int BM                = 32;
    constexpr int BN                = 128;
    constexpr int BK                = 64;
    constexpr int WN                = 16;
    constexpr int MT                = 2;
    constexpr int NT                = 2;
    constexpr int KSUB              = 4;
    constexpr int WARPS_N           = BN / WN;
    constexpr int WARPS             = 8;
    constexpr int THREADS           = WARPS * 32;
    constexpr int SCALE_CACHE_BYTES = 16;

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[2][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[2][BM * BK];
    __shared__ __align__(16) std::uint8_t Sr[2][BM * SCALE_CACHE_BYTES];

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wn   = warp % WARPS_N;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int m0   = static_cast<int>(blockIdx.x) * BM;
    const int n0   = static_cast<int>(blockIdx.y) * BN;
    const int kg   = padded_k / 32;

    float acc_k[MT][NT][4];
    float acc_v[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
#pragma unroll
            for (int ci = 0; ci < 4; ++ci) {
                acc_k[mi][ni][ci] = 0.0f;
                acc_v[mi][ni][ci] = 0.0f;
            }
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
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
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

    auto stage_codes = [&](int p, int kt) {
        const auto* codes = p == 0 ? k_codes : v_codes;
        const int g0      = kt * 2;
#pragma unroll 1
        for (int item = tid; item < BM * (BK / 16); item += THREADS) {
            const int row   = item / (BK / 16);
            const int chunk = item - row * (BK / 16);
            const int grow  = m0 + row;
            auto* dst       = &Cr[p][row * BK + chunk * 16];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                cp_async<16, Cache::cg>(dst, &codes[gi * 32 + chunk * 16]);
            } else {
                const std::int64_t gi = static_cast<std::int64_t>(grow < m ? grow : 0) * kg + g0;
                ninfer::ops::cp_async_zfill<16>(dst, &codes[gi * 32 + chunk * 16],
                                                grow < m ? 16 : 0);
            }
        }
    };

    auto stage_scales = [&](int p, int kt) {
        if ((kt & 3) == 0) {
            const auto* scales = p == 0 ? k_scales : v_scales;
            const int g0       = kt * 2;
            for (int row = tid; row < BM; row += THREADS) {
                const int grow = m0 + row;
                auto* dst      = &Sr[p][row * SCALE_CACHE_BYTES];
                if constexpr (FullTiles) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g0;
                    if (g0 + 8 <= kg) {
                        cp_async<16, Cache::cg>(dst, &scales[gi * 2]);
                    } else {
                        ninfer::ops::cp_async_zfill<16>(dst, &scales[gi * 2], max(0, kg - g0) * 2);
                    }
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

    auto dequant = [&](int p, int kt) {
        const int scale_pair_offset = (kt & 3) * 4;
        const int half              = lane >> 4;
        const int half_lane         = lane & 15;
        for (int row_pair = warp * 2; row_pair < BM; row_pair += WARPS * 2) {
            const int row       = row_pair + half;
            unsigned scale_pair = half_lane == 0
                                      ? *reinterpret_cast<const std::uint32_t*>(
                                            &Sr[p][row * SCALE_CACHE_BYTES + scale_pair_offset])
                                      : 0;
            scale_pair          = __shfl_sync(0xffffffffu, scale_pair, half * 16);
#pragma unroll
            for (int gg = 0; gg < 2; ++gg) {
                const float scale =
                    __half2float(__ushort_as_half((scale_pair >> (gg * 16)) & 0xffffu));
                const int col = gg * 32 + half_lane * 2;
                const std::uint16_t packed =
                    *reinterpret_cast<const std::uint16_t*>(&Cr[p][row * BK + col]);
                const int q0 = static_cast<int>(static_cast<std::int8_t>(packed & 0xffu));
                const int q1 = static_cast<int>(static_cast<std::int8_t>(packed >> 8));
                const __nv_bfloat162 values = __floats2bfloat162_rn(static_cast<float>(q0) * scale,
                                                                    static_cast<float>(q1) * scale);
                store_vec(&As[row * BK + w8g32_swz64(row, col)], values);
            }
        }
    };

    auto mma_pair = [&](int p, int stage) {
        unsigned af[2][MT][4];
        unsigned bf[2][NT][2];
        auto load_fragments = [&](int slot, int ks) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int ar = mi * 16 + a_rowoff;
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
                    if (p == 0) {
                        mma_bf16(acc_k[mi][ni][0], acc_k[mi][ni][1], acc_k[mi][ni][2],
                                 acc_k[mi][ni][3], af[slot][mi][0], af[slot][mi][1],
                                 af[slot][mi][2], af[slot][mi][3], bf[slot][ni][0],
                                 bf[slot][ni][1]);
                    } else {
                        mma_bf16(acc_v[mi][ni][0], acc_v[mi][ni][1], acc_v[mi][ni][2],
                                 acc_v[mi][ni][3], af[slot][mi][0], af[slot][mi][1],
                                 af[slot][mi][2], af[slot][mi][3], bf[slot][ni][0],
                                 bf[slot][ni][1]);
                    }
                }
            }
        }
    };

    const int nkt = padded_k / BK;
    stage_x(0, 0);
    stage_codes(0, 0);
    stage_codes(1, 0);
    stage_scales(0, 0);
    stage_scales(1, 0);
    ninfer::ops::cp_commit();

#pragma unroll 2
    for (int kt = 0; kt < nkt; ++kt) {
        const int stage = kt & 1;
        ninfer::ops::cp_wait<0>();
        __syncthreads();

        dequant(0, kt);
        __syncthreads();
        mma_pair(0, stage);
        __syncthreads();

        dequant(1, kt);
        __syncthreads();

        const int next = kt + 1;
        if (next < nkt) {
            stage_x(next & 1, next);
            stage_codes(0, next);
            stage_codes(1, next);
            stage_scales(0, next);
            stage_scales(1, next);
            ninfer::ops::cp_commit();
        }
        mma_pair(1, stage);
    }

#pragma unroll
    for (int p = 0; p < 2; ++p) {
        auto* out = p == 0 ? k_out : v_out;
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int r0 = m0 + mi * 16 + gid;
            const int r1 = r0 + 8;
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int c0   = n0 + wn * WN + ni * 8 + 2 * lid;
                const int c1   = c0 + 1;
                const float* a = p == 0 ? acc_k[mi][ni] : acc_v[mi][ni];
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
}

} // namespace ninfer::ops::detail
