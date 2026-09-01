#pragma once

// Fused FP8 (E4M3) gate_up projection + SwiGLU on Volta tensor cores, quadpair-split-N form
// (sm_70 only). The FP8 sibling of nvfp4_linear_swiglu_volta_qpn.cuh, and simpler: FP8's scale is
// one BF16 value per output row (not a swizzled per-16-k block plane), so it is read once per
// thread and applied only in the epilogue -- the K loop carries no scale at all, for either half.
//
// Structurally identical to the NVFP4 version otherwise: gate and up are the same K and the same
// activations, so one activation load feeds two independent accumulator sets per 128-byte code
// block, and the grid covers only the gate half. Both row-scale arrays are the same flat N-length
// BF16 array `w.scales`, so the up half's scale pointer is just `scales + kIntermediate` -- no
// swizzle-tile offset to compute the way NVFP4's BlockScaleK16M128x4 plane needed.
//
// Why this exists rather than composing linear() + silu_mul(): the same reason as the NVFP4
// kernel. linear()'s A16 output rounds the projection to BF16 before silu_mul ever sees it, and
// silu(gate) * up compounds that rounding multiplicatively across two independently-rounded
// operands. Untested here directly, but the tolerance function is shared with NVFP4's test
// (linear_swiglu_test_common.cpp's A16 criterion applies to every qtype alike), so the same
// failure mode is expected and this was built directly rather than composed-then-reverted again.

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/volta_mma.cuh"
#include "ops/linear/fp8/fp8_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

inline constexpr int kFp8SwigluIntermediate = 17408; // Fp8MlpGateUpGeometry::kOutputRows / 2
inline constexpr int kFp8SwigluWarps = 4;  // not yet generalized to SPLITK; see fp8_volta_qpn_gemm.cuh
inline constexpr int kFp8SwigluThreads = kFp8SwigluWarps * 32;

// Half the occupancy target plain QPN8 uses at each kTiles: this kernel carries two full
// accumulator sets and two decode-register sets (gate and up) instead of one, and holding the
// same block count target as the single-projection kernel forced severe spilling (184 bytes at
// kTiles=1, sending the kernel from a ~450us op-bench point to ~990us measured in a real round --
// the same occupancy-formula mistake the NVFP4 sibling made and fixed first).
template <int kTiles>
__global__ __launch_bounds__(kFp8SwigluThreads,
                             kTiles == 1 ? 4 : (kTiles == 2 ? 2 : 1)) void fp8_linear_swiglu_volta_qpn_kernel(
    const std::uint8_t* __restrict__ codes, const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, int k, int t, __nv_bfloat16* __restrict__ out) {
    using S             = Fp8VoltaQpnSchedule;
    constexpr int kHalf = kFp8SwigluIntermediate;

    __shared__ float cs_gate[kFp8SwigluWarps][kTiles * S::kRowsPerTile * S::kColsPerCta];
    __shared__ float cs_up[kFp8SwigluWarps][kTiles * S::kRowsPerTile * S::kColsPerCta];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int qp   = (lane >> 2) & 3;
    const int r    = (lane & 3) + ((lane & 16) != 0 ? 4 : 0);

    const int col      = static_cast<int>(blockIdx.x) * S::kColsPerCta + qp * 8 + r;
    const int good     = col < kHalf;
    const int gate_row = good ? col : 0;
    const int up_row   = gate_row + kHalf;

    const int blocks = k / S::kKPerBlock;
    const int bq     = blocks / kFp8SwigluWarps;
    const int b0     = warp * bq;
    const int bend   = (warp == kFp8SwigluWarps - 1) ? blocks : b0 + bq;

    const std::uint8_t* crow_gate = codes + static_cast<std::int64_t>(gate_row) * k;
    const std::uint8_t* crow_up   = codes + static_cast<std::int64_t>(up_row) * k;

    float c_gate[kTiles][8];
    float c_up[kTiles][8];
#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            c_gate[tile][i] = 0.0f;
            c_up[tile][i]   = 0.0f;
        }
    }

    for (int b = b0; b < bend; ++b) {
        const std::uint8_t* pg = crow_gate + static_cast<std::int64_t>(b) * S::kKPerBlock;
        const std::uint8_t* pu = crow_up + static_cast<std::int64_t>(b) * S::kKPerBlock;
        uint4 gw[8];
        uint4 uw[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            gw[e] = __ldg(reinterpret_cast<const uint4*>(pg + 16 * e));
            uw[e] = __ldg(reinterpret_cast<const uint4*>(pu + 16 * e));
        }

#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const std::uint32_t gwords[4] = {gw[e].x, gw[e].y, gw[e].z, gw[e].w};
            const std::uint32_t uwords[4] = {uw[e].x, uw[e].y, uw[e].z, uw[e].w};
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                half2 bg4[4];
                half2 bu4[4];
                fp8_decode_quad(gwords[2 * u], bg4[0], bg4[1]);
                fp8_decode_quad(gwords[2 * u + 1], bg4[2], bg4[3]);
                fp8_decode_quad(uwords[2 * u], bu4[0], bu4[1]);
                fp8_decode_quad(uwords[2 * u + 1], bu4[2], bu4[3]);
                const unsigned* Bg = reinterpret_cast<const unsigned*>(bg4);
                const unsigned* Bu = reinterpret_cast<const unsigned*>(bu4);
                const int kbase    = b * S::kKPerBlock + e * 16 + u * 8;

#pragma unroll
                for (int tile = 0; tile < kTiles; ++tile) {
                    const int row = tile * S::kRowsPerTile + r;
                    half2 a[4];
                    if (row < t) {
                        const __nv_bfloat16* xrow = x + static_cast<std::int64_t>(row) * k + kbase;
                        const uint4 raw            = *reinterpret_cast<const uint4*>(xrow);
                        const auto* src            = reinterpret_cast<const __nv_bfloat16*>(&raw);
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
                    // One activation load feeds both independent projections.
                    volta_mma_qp_n(c_gate[tile], A[0], A[1], Bg[0], Bg[1]);
                    volta_mma_qp_n(c_gate[tile], A[2], A[3], Bg[2], Bg[3]);
                    volta_mma_qp_n(c_up[tile], A[0], A[1], Bu[0], Bu[1]);
                    volta_mma_qp_n(c_up[tile], A[2], A[3], Bu[2], Bu[3]);
                }
            }
        }
    }

#pragma unroll
    for (int tile = 0; tile < kTiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = (i & 2) | ((lane & 16) != 0 ? 4 : 0) | (lane & 1);
            const int cl  = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
            const int e   = (tile * S::kRowsPerTile + row) * S::kColsPerCta + qp * 8 + cl;
            cs_gate[warp][e] = c_gate[tile][i];
            cs_up[warp][e]   = c_up[tile][i];
        }
    }
    __syncthreads(); // the only barrier: cross-warp K reduce, for both accumulator sets at once

    constexpr int kOut = kTiles * S::kRowsPerTile * S::kColsPerCta;
    for (int e = static_cast<int>(threadIdx.x); e < kOut; e += kFp8SwigluThreads) {
        const int row  = e / S::kColsPerCta;
        const int cl   = e % S::kColsPerCta;
        const int ocol = static_cast<int>(blockIdx.x) * S::kColsPerCta + cl;
        if (row < t && ocol < kHalf) {
            float vg = 0.0f;
            float vu = 0.0f;
#pragma unroll
            for (int w = 0; w < kFp8SwigluWarps; ++w) {
                vg += cs_gate[w][e];
                vu += cs_up[w][e];
            }
            // Each half's row scale and the decode's 2^-8, applied once here, matching plain
            // QPN8's epilogue -- the K loop above carries no scale multiply for either half.
            const float scale_gate = __bfloat162float(scales[ocol]) * 256.0f;
            const float scale_up   = __bfloat162float(scales[kHalf + ocol]) * 256.0f;
            out[static_cast<std::int64_t>(row) * kHalf + ocol] =
                __float2bfloat16_rn(silu(vg * scale_gate) * (vu * scale_up));
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
