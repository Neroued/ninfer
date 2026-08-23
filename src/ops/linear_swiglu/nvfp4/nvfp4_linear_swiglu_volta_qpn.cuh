#pragma once

// Fused NVFP4 gate_up projection + SwiGLU on Volta tensor cores, quadpair-split-N form (sm_70
// only). The SwiGLU sibling of nvfp4_volta_qpn_gemm.cuh: same decoder, same fragment maps, same
// per-group loop shape -- what's different is that one CTA now computes *two* independent
// projections (gate and up) of the same output index and combines them with SiLU before the only
// BF16 rounding happens, instead of one.
//
// Why this exists instead of composing linear() + silu_mul(): it was tried first, on both this
// kernel's problem (linear_add, where it's fine) and this one. It failed
// ninfer_linear_swiglu_nvfp4_test by ~0.8% against a ~0.35% tolerance. linear()'s A16 output
// necessarily rounds the projection to BF16 before silu_mul ever sees it; silu(gate) * up
// compounds that rounding multiplicatively across two independently-rounded operands, where
// linear_add's `residual + linear(x,w)` only adds one rounded quantity to another and stays inside
// its (looser, additive) tolerance. Closing the gap means never materializing an intermediate BF16
// gate/up at all -- the accumulator has to stay fp32 from the mma output straight through the SiLU
// multiply, exactly what the existing SIMT nvfp4_linear_swiglu_small_t_kernel already does
// (see nvfp4_linear_swiglu_small_t.cu: one warp's two `parent_rows` are {gate_row, gate_row +
// kIntermediate}, and it combines them post-reduce). This kernel is that same idea on the
// quadpair-split-N tensor-core geometry instead of SIMT.
//
// Structurally: gate and up are non-adjacent N ranges of the *same* K, sharing the *same*
// activations -- so the grid covers only the gate half (kIntermediate columns, half of QPN2's
// plain-Linear grid on this shape), and each CTA runs the K-loop once but decodes and accumulates
// both halves' weights per group, reusing one activation load for both mma sets. Total weight
// bytes read is unchanged from two separate half-grid passes; this just amortizes the activation
// traffic and does the SwiGLU combine before any BF16 round instead of after.
//
// The gate/up row offset is a compile-time constant (kIntermediate = 17408) rather than a runtime
// parameter, matching the existing fused kernel's own specialization to this one registered shape.
// Because kIntermediate is an exact multiple of 128 (17408 = 136*128), the up half's BlockScaleK16
// M128x4 tile index is the gate half's plus a fixed 136 -- computed once, not re-derived per row.

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/volta_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

inline constexpr int kNvfp4SwigluIntermediate = 17408; // Nvfp4MlpGateUpGeometry::kOutputRows / 2
inline constexpr int kNvfp4SwigluMTileOffset  = kNvfp4SwigluIntermediate / 128; // 136, exact

template <int kTiles, int SPLITK, int NACC>
__global__ __launch_bounds__(
    SPLITK * 32, (kTiles == 1 ? 32 : 16) / SPLITK < 1
        ? 1
        : (kTiles == 1 ? 32 : 16) / SPLITK) void nvfp4_linear_swiglu_volta_qpn_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, int k, int t, float inverse_weight_divisor,
    __nv_bfloat16* __restrict__ out) {
    using S             = Nvfp4VoltaQpnSchedule;
    constexpr int kHalf = kNvfp4SwigluIntermediate;

    __shared__ float cs_gate[SPLITK][kTiles * S::kRowsPerTile * S::kColsPerCta];
    __shared__ float cs_up[SPLITK][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col     = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good    = col < kHalf;
    const int gate_row = good ? col : 0;
    const int up_row    = gate_row + kHalf;

    const int scale_tiles = k / S::kChunkK;
    const int stq          = scale_tiles / SPLITK;
    const int st0           = warp * stq;
    const int stend         = (warp == SPLITK - 1) ? scale_tiles : st0 + stq;

    const int m_tile      = gate_row / 128;
    const int row_inner    = gate_row - m_tile * 128;
    const int row_mod32    = row_inner & 31;
    const int row_quartile = row_inner >> 5;
    const std::int64_t scale_row_base_gate = static_cast<std::int64_t>(m_tile) * scale_tiles * 512 +
                                             row_mod32 * 16 + row_quartile * 4;
    // up_row = gate_row + kHalf, and kHalf is an exact multiple of 128, so up's m_tile is exactly
    // the gate m_tile plus kNvfp4SwigluMTileOffset and row_inner is unchanged.
    const std::int64_t scale_row_base_up =
        scale_row_base_gate +
        static_cast<std::int64_t>(kNvfp4SwigluMTileOffset) * scale_tiles * 512;

    const std::uint8_t* crow_gate = codes + static_cast<std::int64_t>(gate_row) * (k / 2);
    const std::uint8_t* crow_up   = codes + static_cast<std::int64_t>(up_row) * (k / 2);

    const half2 rebias  = __float2half2_rn(16384.0f);
    const half2 divisor2 = __float2half2_rn(inverse_weight_divisor * 256.0f);

    float c_gate[kTiles][NACC][8];
    float c_up[kTiles][NACC][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 0; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c_gate[tile][a][i] = 0.0f;
                c_up[tile][a][i]   = 0.0f;
            }
        }
    }

    for (int st = st0; st < stend; ++st) {
        const std::uint8_t* cpg = crow_gate + static_cast<std::int64_t>(st) * S::kCodeBytesPerChunk;
        const std::uint8_t* cpu = crow_up + static_cast<std::int64_t>(st) * S::kCodeBytesPerChunk;
        const uint4 gcode01 = __ldg(reinterpret_cast<const uint4*>(cpg));
        const uint4 gcode23 = __ldg(reinterpret_cast<const uint4*>(cpg + 16));
        const uint4 ucode01 = __ldg(reinterpret_cast<const uint4*>(cpu));
        const uint4 ucode23 = __ldg(reinterpret_cast<const uint4*>(cpu + 16));
        const std::uint32_t gscale_word = *reinterpret_cast<const std::uint32_t*>(
            scales + scale_row_base_gate + static_cast<std::int64_t>(st) * 512);
        const std::uint32_t uscale_word = *reinterpret_cast<const std::uint32_t*>(
            scales + scale_row_base_up + static_cast<std::int64_t>(st) * 512);

#pragma unroll
        for (int g = 0; g < 4; ++g) {
            const std::uint32_t gword_lo = g == 0 ? gcode01.x : g == 1 ? gcode01.z
                                           : g == 2 ? gcode23.x         : gcode23.z;
            const std::uint32_t gword_hi = g == 0 ? gcode01.y : g == 1 ? gcode01.w
                                           : g == 2 ? gcode23.y         : gcode23.w;
            const std::uint32_t uword_lo = g == 0 ? ucode01.x : g == 1 ? ucode01.z
                                           : g == 2 ? ucode23.x         : ucode23.z;
            const std::uint32_t uword_hi = g == 0 ? ucode01.y : g == 1 ? ucode01.w
                                           : g == 2 ? ucode23.y         : ucode23.w;
            const half2 gsc2 = __hmul2(
                nvfp4_decode_e4m3_scale(static_cast<std::uint8_t>(gscale_word >> (8 * g))), divisor2);
            const half2 usc2 = __hmul2(
                nvfp4_decode_e4m3_scale(static_cast<std::uint8_t>(uscale_word >> (8 * g))), divisor2);

            half2 bg[8];
            half2 bu[8];
            nvfp4_decode_e2m1_quad(gword_lo, rebias, *reinterpret_cast<half2(*)[4]>(&bg[0]));
            nvfp4_decode_e2m1_quad(gword_hi, rebias, *reinterpret_cast<half2(*)[4]>(&bg[4]));
            nvfp4_decode_e2m1_quad(uword_lo, rebias, *reinterpret_cast<half2(*)[4]>(&bu[0]));
            nvfp4_decode_e2m1_quad(uword_hi, rebias, *reinterpret_cast<half2(*)[4]>(&bu[4]));
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                bg[j] = __hmul2(bg[j], gsc2);
                bu[j] = __hmul2(bu[j], usc2);
            }
            const unsigned* Bg = reinterpret_cast<const unsigned*>(bg);
            const unsigned* Bu = reinterpret_cast<const unsigned*>(bu);
            const int kbase    = st * S::kChunkK + g * S::kGroupK;

#pragma unroll
            for (int tile = 0; tile < kTiles; ++tile) {
                const int row = tile * S::kRowsPerTile + r;
                half2 a[8];
                if (row < t) {
                    const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(row) * k + kbase;
                    const uint4 raw0 = *reinterpret_cast<const uint4*>(xrow);
                    const uint4 raw1 = *reinterpret_cast<const uint4*>(xrow + 8);
                    const auto* src0 = reinterpret_cast<const __nv_bfloat16*>(&raw0);
                    const auto* src1 = reinterpret_cast<const __nv_bfloat16*>(&raw1);
                    half seq0[8];
                    half seq1[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) { seq0[j] = __float2half(__bfloat162float(src0[j])); }
#pragma unroll
                    for (int j = 0; j < 8; ++j) { seq1[j] = __float2half(__bfloat162float(src1[j])); }
                    nvfp4_stage_pairs(seq0, &a[0]);
                    nvfp4_stage_pairs(seq1, &a[4]);
                } else {
#pragma unroll
                    for (int j = 0; j < 8; ++j) { a[j] = __half2half2(__ushort_as_half(0)); }
                }
                // One activation load feeds both independent projections -- the whole saving
                // over two separate QPN2 passes over this same problem.
                const unsigned* A = reinterpret_cast<const unsigned*>(a);
                volta_mma_qp_n(c_gate[tile][0 % NACC], A[0], A[1], Bg[0], Bg[1]);
                volta_mma_qp_n(c_gate[tile][1 % NACC], A[2], A[3], Bg[2], Bg[3]);
                volta_mma_qp_n(c_gate[tile][2 % NACC], A[4], A[5], Bg[4], Bg[5]);
                volta_mma_qp_n(c_gate[tile][3 % NACC], A[6], A[7], Bg[6], Bg[7]);
                volta_mma_qp_n(c_up[tile][0 % NACC], A[0], A[1], Bu[0], Bu[1]);
                volta_mma_qp_n(c_up[tile][1 % NACC], A[2], A[3], Bu[2], Bu[3]);
                volta_mma_qp_n(c_up[tile][2 % NACC], A[4], A[5], Bu[4], Bu[5]);
                volta_mma_qp_n(c_up[tile][3 % NACC], A[6], A[7], Bu[6], Bu[7]);
            }
        }
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 1; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c_gate[tile][0][i] += c_gate[tile][a][i];
                c_up[tile][0][i] += c_up[tile][a][i];
            }
        }
    }

    // C map (v100-skinny mma8_probe.cu, roles swapped): identical to the plain QPN2 kernel.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            const int e   = (tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl;
            cs_gate[warp][e] = c_gate[tile][0][i];
            cs_up[warp][e]   = c_up[tile][0][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce, for both accumulator sets at once

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += SPLITK * 32) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row < t && ocol < kHalf) {
            float vg = 0.0f;
            float vu = 0.0f;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) {
                vg += cs_gate[w][e];
                vu += cs_up[w][e];
            }
            // silu(gate) * up, entirely in fp32 -- the one BF16 round is the store below, matching
            // the SIMT fused kernel's precision profile exactly (see file header).
            out[static_cast<std::int64_t>(row) * kHalf + ocol] = __float2bfloat16_rn(silu(vg) * vu);
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
