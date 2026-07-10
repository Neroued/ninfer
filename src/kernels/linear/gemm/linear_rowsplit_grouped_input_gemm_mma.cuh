#pragma once

// One-launch grouped Q4/Q5 input projection for the text attention and GDN
// mixers.  Every CTA still owns one homogeneous Q4 or Q5 BMxBN output tile;
// blockIdx.x maps across all jobs so the scheduler can fill the device from the
// combined projection grid without forcing the two codecs into one CTA.

#include "kernels/linear/gemm/linear_rowsplit_gemm_mma.cuh"
#include "qus/core/tensor.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

struct RowsplitGroupedJob {
    const std::uint8_t* codes   = nullptr;
    const std::uint8_t* high    = nullptr;
    const std::uint8_t* scales  = nullptr;
    __nv_bfloat16* out          = nullptr;
    std::int32_t n              = 0;
    std::int32_t out_ld         = 0;
    std::int32_t out_row_offset = 0;
    bool q5                     = false;
};

template <class Cfg, bool FullTiles>
__global__
__launch_bounds__(Cfg::THREADS, Cfg::MIN_BLOCKS) void linear_rowsplit_grouped_input_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x, RowsplitGroupedJob job0, RowsplitGroupedJob job1,
    RowsplitGroupedJob job2, RowsplitGroupedJob job3, std::int32_t k, std::int32_t t,
    std::int32_t padded_k) {
    constexpr int BM   = Cfg::BM;
    constexpr int BN   = Cfg::BN;
    constexpr int BK   = Cfg::BK;
    constexpr int WM   = Cfg::WM;
    constexpr int WN   = Cfg::WN;
    constexpr int MT   = Cfg::MT;
    constexpr int NT   = Cfg::NT;
    constexpr int GPB  = Cfg::GROUPS_PER_BK;
    constexpr int KSUB = BK / 16;
    constexpr int S    = Cfg::STAGES;
    constexpr int SB   = Cfg::SCALE_BYTES;
    static_assert(GPB == 1, "grouped input GEMM requires BK=group_size=64");

    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];
    __shared__ __align__(16) std::uint8_t Cr[S][BM * 32];
    __shared__ __align__(16) std::uint8_t Hr[S][BM * 8];
    __shared__ __align__(16) std::uint8_t Sr[S][BM * SB];

    const int tiles0 = (job0.n + BM - 1) / BM;
    const int tiles1 = (job1.n + BM - 1) / BM;
    const int tiles2 = (job2.n + BM - 1) / BM;
    int tile         = static_cast<int>(blockIdx.x);
    RowsplitGroupedJob job;
    if (tile < tiles0) {
        job = job0;
    } else if ((tile -= tiles0) < tiles1) {
        job = job1;
    } else if ((tile -= tiles1) < tiles2) {
        job = job2;
    } else {
        tile -= tiles2;
        job = job3;
    }

    const int kg   = padded_k >> 6;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::WARPS_N;
    const int wn   = warp % Cfg::WARPS_N;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int m0   = tile * BM;
    const int t0   = static_cast<int>(blockIdx.y) * BN;

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

    const int NKT      = padded_k / BK;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    auto stage_load_x = [&](int stage, int kt) {
        const int k0 = kt * BK;
#pragma unroll 1
        for (int c = tid; c < BN * (BK / 8); c += Cfg::THREADS) {
            const int tl       = c / (BK / 8);
            const int kg8      = c - tl * (BK / 8);
            const int kl       = kg8 * 8;
            const int col      = t0 + tl;
            const int kk       = k0 + kl;
            __nv_bfloat16* dst = &Bs[stage][tl * BK + gemm_swz64(tl, kl)];
            if constexpr (FullTiles) {
                gemm_async_copy_global_to_shared<16, Cfg>(
                    dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else if (col < t && kk + 8 <= k) {
                gemm_async_copy_global_to_shared<16, Cfg>(
                    dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
    };

    auto stage_load_quant = [&](int stage, int kt) {
        const int g = (kt * BK) >> 6;
#pragma unroll 1
        for (int c = tid; c < BM * 2; c += Cfg::THREADS) {
            const int row  = c >> 1;
            const int half = c & 1;
            const int grow = m0 + row;
            auto* dst      = &Cr[stage][row * 32 + half * 16];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_async_copy_global_to_shared<16, Cfg>(dst, &job.codes[gi * 32 + half * 16]);
            } else if (grow < job.n) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                gemm_async_copy_global_to_shared<16, Cfg>(dst, &job.codes[gi * 32 + half * 16]);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
        if (job.q5) {
#pragma unroll 1
            for (int row = tid; row < BM; row += Cfg::THREADS) {
                const int grow = m0 + row;
                auto* dst      = &Hr[stage][row * 8];
                if constexpr (FullTiles) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                    gemm_async_copy_global_to_shared<8, Cfg>(dst, &job.high[gi * 8]);
                } else if (grow < job.n) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + g;
                    gemm_async_copy_global_to_shared<8, Cfg>(dst, &job.high[gi * 8]);
                } else {
                    *reinterpret_cast<std::uint64_t*>(dst) = 0;
                }
            }
        }
#pragma unroll 1
        for (int row = tid; row < BM; row += Cfg::THREADS) {
            const int grow = m0 + row;
            auto* dst      = &Sr[stage][row * SB];
            if constexpr (FullTiles) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    if (aligned_g + 1 < kg) {
                        gemm_async_copy_global_to_shared<4, Cfg>(dst, &job.scales[aligned_gi * 2]);
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
                        *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                    }
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
                }
            } else if (grow < job.n) {
                const int aligned_g           = g & ~1;
                const std::int64_t gi         = static_cast<std::int64_t>(grow) * kg + g;
                const std::int64_t aligned_gi = static_cast<std::int64_t>(grow) * kg + aligned_g;
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    if (aligned_g + 1 < kg) {
                        gemm_async_copy_global_to_shared<4, Cfg>(dst, &job.scales[aligned_gi * 2]);
                    } else {
                        *reinterpret_cast<std::uint16_t*>(dst) =
                            *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
                        *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                    }
                } else {
                    *reinterpret_cast<std::uint16_t*>(dst) =
                        *reinterpret_cast<const std::uint16_t*>(&job.scales[gi * 2]);
                }
            } else {
                *reinterpret_cast<std::uint16_t*>(dst) = 0;
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    *reinterpret_cast<std::uint16_t*>(dst + 2) = 0;
                }
            }
        }
    };

    auto stage_load = [&](int stage, int kt) {
        stage_load_x(stage, kt);
        stage_load_quant(stage, kt);
    };

    auto dequant_to_As = [&](int stage, int kt) {
        const int scale_off = ((kt * BK >> 6) & 1) * 2;
        for (int row = warp; row < BM; row += Cfg::WARPS) {
            __nv_bfloat162 w;
            if (job.q5) {
                if constexpr (Cfg::SCALE_PAIR_LOAD) {
                    w = Q5Codec::load_pair_bf162_scale_ptr(
                        Cr[stage], Hr[stage], &Sr[stage][row * SB + scale_off], row, lane);
                } else {
                    w = Q5Codec::load_pair_bf162(Cr[stage], Hr[stage], Sr[stage], row, lane);
                }
            } else if constexpr (Cfg::SCALE_PAIR_LOAD) {
                w = Q4Codec::load_pair_bf162_scale_ptr(Cr[stage], nullptr,
                                                       &Sr[stage][row * SB + scale_off], row, lane);
            } else {
                w = Q4Codec::load_pair_bf162(Cr[stage], nullptr, Sr[stage], row, lane);
            }
            const int sc                                           = gemm_swz64(row, 2 * lane);
            *reinterpret_cast<__nv_bfloat162*>(&As[row * BK + sc]) = w;
        }
    };

#pragma unroll
    for (int s = 0; s < S; ++s) {
        if (s < NKT) { stage_load(s, s); }
        qus::kernels::async_copy_commit();
    }

    for (int it = 0; it < NKT; ++it) {
        const int stage = it % S;
        qus::kernels::async_copy_wait<S - 1>();
        __syncthreads();
        dequant_to_As(stage, it);
        __syncthreads();

        unsigned af[MT][4];
        unsigned bf[NT][2];
#pragma unroll
        for (int ki = 0; ki < KSUB; ++ki) {
            const int ks = ki * 16;
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int arow = wm * WM + mi * 16 + a_rowoff;
                gemm_ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                 gemm_smem_addr(&As[arow * BK + gemm_swz64(arow, ks + a_coloff)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int brow = wn * WN + ni * 8 + b_rin;
                gemm_ldmatrix_x2(
                    bf[ni][0], bf[ni][1],
                    gemm_smem_addr(&Bs[stage][brow * BK + gemm_swz64(brow, ks + b_koff)]));
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    gemm_mma_m16n8k16_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2],
                                           acc[mi][ni][3], af[mi][0], af[mi][1], af[mi][2],
                                           af[mi][3], bf[ni][0], bf[ni][1]);
                }
            }
        }
        __syncthreads();
        const int next = it + S;
        if (next < NKT) { stage_load(stage, next); }
        qus::kernels::async_copy_commit();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int r0 = m0 + wm * WM + mi * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int cc0 = t0 + wn * WN + ni * 8 + 2 * lid;
            const int cc1 = cc0 + 1;
            auto store    = [&](int col, int row, float value) {
                job.out[static_cast<std::int64_t>(col) * job.out_ld + job.out_row_offset + row] =
                    __float2bfloat16_rn(value);
            };
            if constexpr (FullTiles) {
                store(cc0, r0, acc[mi][ni][0]);
                store(cc1, r0, acc[mi][ni][1]);
                store(cc0, r1, acc[mi][ni][2]);
                store(cc1, r1, acc[mi][ni][3]);
            } else {
                if (r0 < job.n) {
                    if (cc0 < t) { store(cc0, r0, acc[mi][ni][0]); }
                    if (cc1 < t) { store(cc1, r0, acc[mi][ni][1]); }
                }
                if (r1 < job.n) {
                    if (cc0 < t) { store(cc0, r1, acc[mi][ni][2]); }
                    if (cc1 < t) { store(cc1, r1, acc[mi][ni][3]); }
                }
            }
        }
    }
}

void linear_rowsplit_attn_input_grouped_mma_launch(const Tensor& x, const Weight& q_weight,
                                                   const Weight& gate_weight,
                                                   const Weight& k_weight, const Weight& v_weight,
                                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                                   cudaStream_t stream);

void linear_rowsplit_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                                  const Weight& v_weight, Tensor& qkv,
                                                  cudaStream_t stream);

} // namespace qus::kernels::detail
