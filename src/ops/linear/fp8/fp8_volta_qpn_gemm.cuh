#pragma once

// Fused-dequant FP8 (E4M3) x BF16 GEMM on Volta tensor cores, quadpair-split-N form (sm_70 only).
//
// The third sibling of q4_volta_qpn_gemm.cuh and w8_volta_qpn_gemm.cuh, and the simplest of the
// three. Same geometry, same fragment maps, same one-barrier structure: the four quadpairs of a
// warp split N, share a single 8x4 activation tile, and the CTA's four warps split K.
//
// It exists because the FP8 A16 SIMT family decays with T where the groupwise routes do not. The
// QPN mapping keeps the curve step-flat by feeding an eight-row tile from one weight stream.
//
// Parity is not the target. FP8_E4M3FN_ROW_BF16S carries 8 bits per weight plus one BF16 per
// output row (~8.0 bits/weight) against W8G32_F16S's 8 plus an FP16 per 32 (8.5), so this kernel
// moving weights as efficiently as its W8 sibling should read about 6% *faster* than the W8
// numbers above.
//
// Two FP8 specifics, and both make it cheaper than the W8 sibling rather than dearer:
//
//   - The scale is per output row, not per group of 32. It is loaded once before the K loop and
//     applied in the epilogue, so the inner loop carries no scale multiply at all -- the W8
//     sibling pays an __hmul2 per four k, and a blocked scale load per 128 bytes of codes to keep
//     its 79 MB scale plane off DRAM. There is no scale plane here worth the name: 2 bytes per
//     output row.
//   - The decode is a shift, not a magic-number identity. E4M3 is S EEEE MMM, so placing the sign
//     at fp16 bit 15 and the seven exponent+mantissa bits at fp16 bits 7..13 reproduces the value
//     scaled by 2^-8 -- fp16's exponent bias is 15 against E4M3's 7. That 1/256 is folded into the
//     epilogue's row scale, so it costs nothing either.
//
// The mask below is 0x3F80, not 0x7F80. With the wider mask bit 14 catches the sign, which lands
// in the fp16 exponent: every negative weight overflows to inf and accumulates to NaN. This is
// v100-skinny's bug (kernels/skinny_kernels.cu, fp8x8_to_half2x4_fast), recorded there and not
// re-derived here. Their shift was verified exhaustively against PyTorch on all 256 byte patterns:
// exact for every finite E4M3 value including all 14 denormals, with only the two NaN encodings
// mapping to +-480, which weights never contain.

#include "core/device.h"
#include "ops/common/volta_mma.cuh"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Fp8VoltaQpnSchedule {
    static constexpr int kColsPerCta  = 32; // output rows per CTA (mma's N axis)
    static constexpr int kRowsPerTile = 8;  // tokens per A tile (mma's M axis)
    static constexpr int kKPerBlock   = 128; // code bytes a lane reads before consuming any
};

// Four adjacent E4M3 codes -> two half2 of adjacent k, each value carrying a 2^-8 factor.
//
// The permute puts code j at byte 0 and code j+1 at byte 2 of a word, so one shift/mask pair
// builds both fp16 lanes at once. The bytes it leaves in lanes 1 and 3 are garbage on purpose:
// after the shifts they land outside both masks, so zeroing them would be wasted work.
__device__ __forceinline__ void fp8_decode_quad(std::uint32_t word, half2& lo, half2& hi) {
    constexpr std::uint32_t kSign = 0x80008000u;
    constexpr std::uint32_t kExpM = 0x3F803F80u;
    const std::uint32_t p0        = __byte_perm(word, word, 0x0110); // [b0, b1, b1, b0]
    const std::uint32_t p1        = __byte_perm(word, word, 0x2332); // [b2, b3, b3, b2]
    const std::uint32_t v0        = ((p0 << 8) & kSign) | ((p0 << 7) & kExpM);
    const std::uint32_t v1        = ((p1 << 8) & kSign) | ((p1 << 7) & kExpM);
    lo                            = *reinterpret_cast<const half2*>(&v0);
    hi                            = *reinterpret_cast<const half2*>(&v1);
}

// `kTiles` is the number of 8-row A tiles, so T <= 8 * kTiles. Each tile adds eight float
// accumulators and an A fragment per lane, so the resident-CTA target has to come down with it or
// the extra tiles buy spills instead of rows. `SPLITK` (warps per CTA splitting K, generalizing
// the fixed 4 this kernel shipped with) and `NACC` (independent accumulator fragments the two
// per-block mma slices round-robin into) are v100-skinny's generation-2 knobs -- see
// nvfp4_volta_qpn_gemm.cuh, which got them first; this is the same pattern applied to the
// simpler single-projection kernel. The shared reduce buffer is SPLITK * kTiles * 256 floats, so
// SPLITK=16 at kTiles=4 is never instantiated (64 KB, over Volta's 48 KB static limit).
template <int kTiles, int SPLITK, int NACC, class OutputPolicy>
__global__ __launch_bounds__(
    SPLITK * 32, (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK < 1
        ? 1
        : (kTiles == 1 ? 32 : kTiles == 2 ? 16 : 4) / SPLITK) void fp8_volta_qpn_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, int n, int k, int t, OutputPolicy output) {
    using S = Fp8VoltaQpnSchedule;

    __shared__ float cs[SPLITK][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    // Quadpair index, and this lane's position inside it. `r` is both the A row (token) and the
    // B column local to the quadpair.
    const int qp = (lane >> 2) & 3;
    const int r  = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col  = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good = col < n;

    // The CTA's SPLITK warps split K in whole 128-byte blocks so each lane still consumes a full
    // line.
    const int blocks = k / S::kKPerBlock;
    const int bq     = blocks / SPLITK;
    const int b0     = warp * bq;
    const int bend   = (warp == SPLITK - 1) ? blocks : b0 + bq;

    const std::uint8_t* crow = codes + static_cast<std::int64_t>(good ? col : 0) * k;

    float c[kTiles][NACC][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 0; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][a][i] = 0.0f; }
        }
    }

    for (int b = b0; b < bend; ++b) {
        // One 128-byte line per lane per iteration: 128 adjacent k, eight uint4 loads.
        const std::uint8_t* p = crow + static_cast<std::int64_t>(b) * S::kKPerBlock;
        uint4 cw[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) { cw[e] = __ldg(reinterpret_cast<const uint4*>(p + 16 * e)); }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const std::uint32_t words[4] = {cw[e].x, cw[e].y, cw[e].z, cw[e].w};
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                // Two words = 8 adjacent k = the four half2 that two mma slices consume.
                half2 b4[4];
                fp8_decode_quad(words[2 * u], b4[0], b4[1]);
                fp8_decode_quad(words[2 * u + 1], b4[2], b4[3]);
                const unsigned* B = reinterpret_cast<const unsigned*>(b4);
                const int kbase   = b * S::kKPerBlock + e * 16 + u * 8;

#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    const int row = tile * S::kRowsPerTile + r;
                    half2 a[4];
                    if (row < t) {
                        const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(row) * k + kbase;
                        const uint4 raw           = *reinterpret_cast<const uint4*>(xrow);
                        const auto* src           = reinterpret_cast<const __nv_bfloat16*>(&raw);
                        __half tmp[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            tmp[j] = __float2half(__bfloat162float(src[j]));
                        }
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            a[j] = *reinterpret_cast<const half2*>(tmp + 2 * j);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < 4; ++j) { a[j] = __half2half2(__ushort_as_half(0)); }
                    }
                    const unsigned* A = reinterpret_cast<const unsigned*>(a);
                    volta_mma_qp_n(c[tile][0 % NACC], A[0], A[1], B[0], B[1]); // k slice 0
                    volta_mma_qp_n(c[tile][1 % NACC], A[2], A[3], B[2], B[3]); // k slice 1
                }
            }
        }
    }

    // Fold the independent accumulator chains together before the cross-warp reduce.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int a = 1; a < NACC; ++a) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { c[tile][0][i] += c[tile][a][i]; }
        }
    }

    // C map (v100-skinny mma8_probe.cu, roles swapped); see the Q4 sibling.
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            cs[warp][(tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl] = c[tile][0][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += SPLITK * 32) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row < t && ocol < n) {
            float v = 0.0f;
#pragma unroll
            for (int w = 0; w < SPLITK; ++w) { v += cs[w][e]; }
            // Row scale and the decode's 2^-8, together, once per output element. The store goes
            // through the caller's policy so the fused attention projections can scatter straight
            // into their q/gate/k/v (or qkv/z) planes instead of a contiguous buffer they would
            // then have to split.
            const float scale = __bfloat162float(scales[ocol]) * 256.0f;
            output.store(ocol, row, v * scale);
        }
    }
}

// Shared launcher. Every FP8 consumer -- plain Linear, the attention projections, the GDN input
// projection -- differs only in where the epilogue puts its results, so they share one kernel and
// supply their own output policy.
template <class OutputPolicy>
void launch_fp8_volta_qpn_with_output(const Tensor& x, const Weight& w, OutputPolicy output,
                                      std::int32_t n, cudaStream_t stream) {
    using S              = Fp8VoltaQpnSchedule;
    const std::int32_t k = x.ne[0];
    const std::int32_t t = x.ne[1];

    const dim3 grid(static_cast<unsigned>((n + S::kColsPerCta - 1) / S::kColsPerCta));
    const auto* codes  = static_cast<const std::uint8_t*>(w.qdata);
    const auto* scales = static_cast<const __nv_bfloat16*>(w.scales);
    const auto* xd     = static_cast<const __nv_bfloat16*>(x.data);
    // Generation-2 winners from a private sweep (bench/ops/fp8_qpn8_splitk_sweep.cu, deleted):
    // SPLITK8 NACC1 wins at every kTiles on attn input, GDN input, and the 17408-K residual shape
    // (1.08-1.55x over SPLITK4). The 6144-K residual shape is the one exception -- SPLITK16 wins
    // there at kTiles=1 (458.6 vs 404.2 GB/s at T=4) because k/128=48 leaves it more headroom than
    // the 5120/17408-K shapes, where k/128=40/136 aren't even divisible by 16. NACC=2 loses
    // everywhere, matching the NVFP4 SwiGLU kernel's finding: extra accumulator chains just cost
    // registers on a kernel that is already issue/DRAM-bound, not dependency-bound.
    const bool wide_k_headroom = k == 6144;
    if (t <= S::kRowsPerTile) {
        if (wide_k_headroom) {
            fp8_volta_qpn_gemm_kernel<1, 16, 1><<<grid, 16 * 32, 0, stream>>>(codes, scales, xd, n,
                                                                              k, t, output);
        } else {
            fp8_volta_qpn_gemm_kernel<1, 8, 1><<<grid, 8 * 32, 0, stream>>>(codes, scales, xd, n, k,
                                                                            t, output);
        }
    } else if (t <= 2 * S::kRowsPerTile) {
        fp8_volta_qpn_gemm_kernel<2, 8, 1><<<grid, 8 * 32, 0, stream>>>(codes, scales, xd, n, k, t,
                                                                        output);
    } else {
        fp8_volta_qpn_gemm_kernel<4, 8, 1><<<grid, 8 * 32, 0, stream>>>(codes, scales, xd, n, k, t,
                                                                        output);
    }
    CUDA_CHECK(cudaGetLastError());
}

#endif // sm_70

} // namespace ninfer::ops::detail
