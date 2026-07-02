#pragma once

// LargeT tensor-core GEMM: out[N,T] = W[N,K] . x[K,T], W is Q4/Q5/Q6 row-split,
// x/out bf16. bf16 mma.sync (m16n8k16, fp32 accumulate) with the low-bit weights
// dequantized on-chip from the existing row-split layout (single VRAM copy; no
// repack). Dequant math is identical to Codec::load_pair / the reference.
//
// Numerics: bf16 operands (A16-native, 2x the tf32 throughput). This rounds the
// dequantized weight to bf16; the fp64 golden keeps it in fp32, so the tensor-core
// path is judged by a normwise criterion (Tolerance::linear_tc, rel_l2 <= 4e-3),
// per docs/l1-op-test-standard.md.
//
// Architecture (P2 pushdown): a tiled GEMM behind a compile-time GemmCfg. Each
// block owns a BM(rows) x BN(tokens) output tile and contracts K in BK=64 steps
// (one quant group per step, so the fp16 group scale is loaded once per row per
// K-tile, from L1). Per K-step:
//   * x (the mma B operand) is staged once into shared bf16 and reused by every
//     warp in the block (the correctness-first kernel re-read x per element per
//     warp -- the dominant waste);
//   * W (the mma A operand) is dequantized once into a shared bf16 tile and
//     reused across the whole token tile;
//   * each warp accumulates a WM x WN register tile of m16n8k16 mmas, reusing each
//     loaded A fragment across the WN token subtiles and each B fragment across
//     the WM row subtiles.
// Fragments are assembled from the shared bf16 tiles with the documented
// m16n8k16 lane layout (a0..a3 = the four 8x8 quadrants of the 16x16 A tile;
// b0/b1 = the two k-halves of the 16x8 B tile). ldmatrix, cp.async pipelining,
// and the shared-staged epilogue are layered on top during the ncu tuning phase.

#include "kernels/linear/codec/linear_codec.cuh"
#include "kernels/kernel/gdn_common.cuh" // cp.async helpers

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

// High-plane bytes per (row, 64-group) for each codec (0 = no high plane). Used
// to size the staged shared quant buffers; the dequant math itself stays in
// Codec::load_pair.
template <class C>
struct gemm_high_bytes {
    static constexpr int value = 0;
};
template <>
struct gemm_high_bytes<Q5Codec> {
    static constexpr int value = 8;
};
template <>
struct gemm_high_bytes<Q6Codec> {
    static constexpr int value = 16;
};

// Compile-time tile configuration. BK is fixed to the 64-wide quant group so a
// K-step consumes exactly one group per row (single scale load). The block tile
// BM x BN is partitioned into WARPS_M x WARPS_N warp tiles of WM x WN.
template <int BM_, int BN_, int BK_, int WM_, int WN_, int STAGES_, int MIN_BLOCKS_ = 1,
          bool FRAG_DBUF_ = true>
struct GemmCfg {
    static constexpr int BM         = BM_;
    static constexpr int BN         = BN_;
    static constexpr int BK         = BK_;
    static constexpr int WM         = WM_;
    static constexpr int WN         = WN_;
    static constexpr int STAGES     = STAGES_;
    static constexpr int MIN_BLOCKS = MIN_BLOCKS_;
    static constexpr int WARPS_M = BM_ / WM_;
    static constexpr int WARPS_N = BN_ / WN_;
    static constexpr int WARPS   = WARPS_M * WARPS_N;
    static constexpr int THREADS = WARPS * 32;
    static constexpr int MT      = WM_ / 16; // m16 subtiles per warp
    static constexpr int NT      = WN_ / 8;  // n8 subtiles per warp
    static constexpr bool FRAG_DBUF = FRAG_DBUF_;

    static constexpr int GROUPS_PER_BK = BK_ / 64; // quant groups contracted per K-tile

    // Conservative shared estimate: single As (bf16) + STAGES x-tiles (bf16) +
    // STAGES raw quant (codes 32B + high <=16B + scale 2B per group). Uses the
    // Q6 high plane as the upper bound so all codecs fit the static 48KB budget.
    static constexpr int smem_est_bytes =
        (BM_ * BK_) * 2 + STAGES_ * (BN_ * BK_) * 2 + STAGES_ * (BM_ * GROUPS_PER_BK * (32 + 16 + 2));

    static_assert(BK_ % 64 == 0, "GemmCfg: BK must be a multiple of the 64-wide quant group");
    static_assert(STAGES_ >= 2, "GemmCfg: the cp.async pipeline requires STAGES >= 2");
    static_assert(BM_ % WM_ == 0 && BN_ % WN_ == 0,
                  "GemmCfg: block tile must divide into warp tiles");
    static_assert(WM_ % 16 == 0 && WN_ % 8 == 0,
                  "GemmCfg: warp tile must be m16 x n8 multiples");
    static_assert(WARPS >= 1 && THREADS <= 1024, "GemmCfg: thread count out of range");
    static_assert(smem_est_bytes <= 48 * 1024, "GemmCfg: staged shared exceeds the 48KB budget");
};

__device__ __forceinline__ void gemm_mma_m16n8k16_bf16(float& c0, float& c1, float& c2, float& c3,
                                                       unsigned a0, unsigned a1, unsigned a2,
                                                       unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned gemm_smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// XOR swizzle for a [rows][64] bf16 tile: permute the eight 16-byte (8-bf16) column
// groups by (row & 7), preserving within-group order. ldmatrix reads eight
// consecutive rows at a fixed 8-wide column group; unswizzled they collide on the
// same bank (row stride 64 bf16 = 32 banks). With this permutation the eight rows
// land on eight distinct bank groups (conflict-free). Applied identically to the
// dequant write and both ldmatrix reads so it is transparent to the math.
__device__ __forceinline__ int gemm_swz64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

// ldmatrix.x4: load a 16x16 bf16 A tile as the four 8x8 quadrants (a0..a3 =
// Q0=[r0-7,c0-7], Q1=[r8-15,c0-7], Q2=[r0-7,c8-15], Q3=[r8-15,c8-15]) directly
// into the m16n8k16 A-fragment registers. Natural {a0,a1,a2,a3} order to match
// the mma fragment layout (see the correctness-first per-element mapping).
__device__ __forceinline__ void gemm_ldmatrix_x4(unsigned& a0, unsigned& a1, unsigned& a2,
                                                 unsigned& a3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
                 : "r"(addr));
}

// ldmatrix.x2: load a 16(k)x8(n) bf16 B tile (stored [n][k], k-contiguous) as the
// two k-halves (b0 = k0-7, b1 = k8-15) into the m16n8k16 B-fragment registers.
__device__ __forceinline__ void gemm_ldmatrix_x2(unsigned& b0, unsigned& b1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1)
                 : "r"(addr));
}

template <class Codec, class Cfg, bool FullTiles>
__global__ __launch_bounds__(Cfg::THREADS, Cfg::MIN_BLOCKS) void linear_rowsplit_gemm_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t n, std::int32_t k, std::int32_t t,
    std::int32_t padded_k) {
    constexpr int BM   = Cfg::BM;
    constexpr int BN   = Cfg::BN;
    constexpr int BK   = Cfg::BK;
    constexpr int WM   = Cfg::WM;
    constexpr int WN   = Cfg::WN;
    constexpr int MT   = Cfg::MT;
    constexpr int NT   = Cfg::NT;
    constexpr int GPB  = Cfg::GROUPS_PER_BK; // quant groups per K-tile
    constexpr int KSUB = BK / 16;            // k16 substeps per K-tile
    constexpr int S    = Cfg::STAGES;        // cp.async pipeline depth (>=2)
    constexpr int HB   = gemm_high_bytes<Codec>::value;
    constexpr int HBS  = HB > 0 ? HB : 1; // avoid a zero-size shared array for Q4

    // As holds the dequantized weight tile for the current stage; Bs/Cr/Hr/Sr are
    // S-deep so cp.async can prefetch the next K-tile's raw quant + x while the
    // current tile is dequantized and multiplied.
    __shared__ __align__(16) __nv_bfloat16 As[BM * BK];         // dequantized W [BM][BK]
    __shared__ __align__(16) __nv_bfloat16 Bs[S][BN * BK];      // staged x      [BN][BK]
    __shared__ __align__(16) std::uint8_t  Cr[S][BM * GPB * 32];  // nibble codes
    __shared__ __align__(16) std::uint8_t  Hr[S][BM * GPB * HBS]; // high plane
    __shared__ __align__(16) std::uint8_t  Sr[S][BM * GPB * 2];   // fp16 scales

    const int kg   = padded_k >> 6;
    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Cfg::WARPS_N; // warp row within the block tile
    const int wn   = warp % Cfg::WARPS_N; // warp col within the block tile
    const int gid  = lane >> 2;           // 0..7 : mma M-row / B N-col
    const int lid  = lane & 3;            // 0..3 : mma K / C N-col

    const int m0 = static_cast<int>(blockIdx.x) * BM; // first output row (N dim)
    const int t0 = static_cast<int>(blockIdx.y) * BN; // first token (T dim)

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

    const int NKT = padded_k / BK; // number of K-tiles

    // ldmatrix fragment addressing (constant across the K-loop).
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3); // +8 for Q1/Q3
    const int a_coloff = (a_mat >> 1) << 3;          // +8 for Q2/Q3
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3; // +8 for the k8-15 half

    // Prefetch K-tile `kt`'s raw quant (codes/high) + x into stage buffer `stage`
    // with cp.async (coalesced 16B); scales are 2B/group and read scalar. No math.
    auto stage_load = [&](int stage, int kt) {
        const int k0 = kt * BK;
        const int g  = k0 >> 6;
#pragma unroll 1
        for (int c = tid; c < BN * (BK / 8); c += Cfg::THREADS) {
            const int      tl  = c / (BK / 8);
            const int      kg8 = c - tl * (BK / 8);
            const int      kl  = kg8 * 8;
            const int      col = t0 + tl;
            const int      kk  = k0 + kl;
            __nv_bfloat16* dst = &Bs[stage][tl * BK + gemm_swz64(tl, kl)];
            if constexpr (FullTiles) {
                qus::kernels::async_copy_global_to_shared<16>(
                    dst, &x[static_cast<std::int64_t>(col) * k + kk]);
            } else {
                if (col < t && kk + 8 <= k) {
                    qus::kernels::async_copy_global_to_shared<16>(
                        dst, &x[static_cast<std::int64_t>(col) * k + kk]);
                } else {
                    *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
                }
            }
        }
#pragma unroll 1
        for (int c = tid; c < BM * GPB * 2; c += Cfg::THREADS) {
            const int     rg   = c >> 1;
            const int     half = c & 1;
            const int     row  = rg / GPB;
            const int     gg   = rg - row * GPB;
            const int     grow = m0 + row;
            std::uint8_t* dst  = &Cr[stage][rg * 32 + half * 16];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + (g + gg);
                qus::kernels::async_copy_global_to_shared<16>(dst, &codes[gi * 32 + half * 16]);
            } else {
                if (grow < n) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + (g + gg);
                    qus::kernels::async_copy_global_to_shared<16>(dst, &codes[gi * 32 + half * 16]);
                } else {
                    *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
                }
            }
        }
        if constexpr (HB > 0) {
#pragma unroll 1
            for (int rg = tid; rg < BM * GPB; rg += Cfg::THREADS) {
                const int     row  = rg / GPB;
                const int     gg   = rg - row * GPB;
                const int     grow = m0 + row;
                std::uint8_t* dst  = &Hr[stage][rg * HB];
                if constexpr (FullTiles) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + (g + gg);
                    qus::kernels::async_copy_global_to_shared<HB>(dst, &high[gi * HB]);
                } else {
                    if (grow < n) {
                        const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + (g + gg);
                        qus::kernels::async_copy_global_to_shared<HB>(dst, &high[gi * HB]);
                    } else {
#pragma unroll
                        for (int b = 0; b < HB; ++b) { dst[b] = 0; }
                    }
                }
            }
        }
#pragma unroll 1
        for (int rg = tid; rg < BM * GPB; rg += Cfg::THREADS) {
            const int     row  = rg / GPB;
            const int     gg   = rg - row * GPB;
            const int     grow = m0 + row;
            std::uint8_t* dst  = &Sr[stage][rg * 2];
            if constexpr (FullTiles) {
                const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + (g + gg);
                dst[0]                = scales[gi * 2];
                dst[1]                = scales[gi * 2 + 1];
            } else {
                if (grow < n) {
                    const std::int64_t gi = static_cast<std::int64_t>(grow) * kg + (g + gg);
                    dst[0]                = scales[gi * 2];
                    dst[1]                = scales[gi * 2 + 1];
                } else {
                    dst[0] = 0;
                    dst[1] = 0;
                }
            }
        }
    };

    // Dequant staged quant of buffer `stage` into As (swizzled), warp-per-row.
    auto dequant_to_As = [&](int stage) {
        for (int row = warp; row < BM; row += Cfg::WARPS) {
            __nv_bfloat16* dst = &As[row * BK];
#pragma unroll
            for (int gg = 0; gg < GPB; ++gg) {
                float w0 = 0.0f;
                float w1 = 0.0f;
                Codec::load_pair(Cr[stage], Hr[stage], Sr[stage], row * GPB + gg, lane, w0, w1);
                const int sc = gemm_swz64(row, gg * 64 + 2 * lane);
                dst[sc]      = __float2bfloat16(w0);
                dst[sc + 1]  = __float2bfloat16(w1);
            }
        }
    };

    // Prologue: launch the first S tiles' async loads.
#pragma unroll
    for (int s = 0; s < S; ++s) {
        if (s < NKT) { stage_load(s, s); }
        qus::kernels::async_copy_commit();
    }

    for (int it = 0; it < NKT; ++it) {
        const int stage = it % S;
        qus::kernels::async_copy_wait<S - 1>();
        __syncthreads();

        dequant_to_As(stage);
        __syncthreads();

        auto load_frag_set = [&](int ks, unsigned (&af)[MT][4], unsigned (&bf)[NT][2]) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int arow = wm * WM + mi * 16 + a_rowoff;
                const int acol = ks + a_coloff;
                gemm_ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                                 gemm_smem_addr(&As[arow * BK + gemm_swz64(arow, acol)]));
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int brow = wn * WN + ni * 8 + b_rin;
                const int bcol = ks + b_koff;
                gemm_ldmatrix_x2(bf[ni][0], bf[ni][1],
                                 gemm_smem_addr(&Bs[stage][brow * BK + gemm_swz64(brow, bcol)]));
            }
        };

        if constexpr (Cfg::FRAG_DBUF) {
            unsigned af[2][MT][4];
            unsigned bf[2][NT][2];
            load_frag_set(0, af[0], bf[0]);
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                const int cur = ki & 1;
                const int nxt = (ki + 1) & 1;
                if (ki + 1 < KSUB) { load_frag_set((ki + 1) * 16, af[nxt], bf[nxt]); }
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        gemm_mma_m16n8k16_bf16(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2],
                                               acc[mi][ni][3], af[cur][mi][0], af[cur][mi][1],
                                               af[cur][mi][2], af[cur][mi][3], bf[cur][ni][0],
                                               bf[cur][ni][1]);
                    }
                }
            }
        } else {
            unsigned af[MT][4];
            unsigned bf[NT][2];
#pragma unroll
            for (int ki = 0; ki < KSUB; ++ki) {
                load_frag_set(ki * 16, af, bf);
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
        }
        __syncthreads();

        const int itp = it + S;
        if (itp < NKT) { stage_load(stage, itp); }
        qus::kernels::async_copy_commit();
    }

    // Epilogue: C lane layout c0={gid,2lid} c1={gid,2lid+1} c2={gid+8,2lid} c3={gid+8,2lid+1}.
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int r0 = m0 + wm * WM + mi * 16 + gid;
        const int r1 = r0 + 8;
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int    cc0 = t0 + wn * WN + ni * 8 + 2 * lid;
            const int    cc1 = cc0 + 1;
            const float* a   = acc[mi][ni];
            if constexpr (FullTiles) {
                out[static_cast<std::int64_t>(cc0) * n + r0] = __float2bfloat16(a[0]);
                out[static_cast<std::int64_t>(cc1) * n + r0] = __float2bfloat16(a[1]);
                out[static_cast<std::int64_t>(cc0) * n + r1] = __float2bfloat16(a[2]);
                out[static_cast<std::int64_t>(cc1) * n + r1] = __float2bfloat16(a[3]);
            } else {
                if (r0 < n) {
                    if (cc0 < t) { out[static_cast<std::int64_t>(cc0) * n + r0] = __float2bfloat16(a[0]); }
                    if (cc1 < t) { out[static_cast<std::int64_t>(cc1) * n + r0] = __float2bfloat16(a[1]); }
                }
                if (r1 < n) {
                    if (cc0 < t) { out[static_cast<std::int64_t>(cc0) * n + r1] = __float2bfloat16(a[2]); }
                    if (cc1 < t) { out[static_cast<std::int64_t>(cc1) * n + r1] = __float2bfloat16(a[3]); }
                }
            }
        }
    }
}

} // namespace qus::kernels::detail
