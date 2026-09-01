#pragma once

// Fused-dequant NVFP4 x BF16 GEMM on Volta tensor cores (mma.sync.m8n8k4), sm_70 only -- the
// wide-T companion to nvfp4_volta_qpn_gemm.cuh, and the piece that was missing for prefill.
//
// Every QPN kernel this port ships (Q4/W8/FP8/NVFP4) is the *quadpair* assignment of
// mma.sync.m8n8k4: T on the 8-row B axis, N on the 32-row A axis via four quadpairs, one
// activation tile shared by all four. That is the right shape when T is small -- it amortizes
// weight decode across many output columns. At prefill's T=2048 it is the wrong shape: QPN2
// dispatches it as 64 separate 32-token launches (nvfp4_dispatch.cpp chunks at
// kNvfp4VoltaQpnMaxTokens=32), so the weight is decoded and streamed *64 times* instead of once.
// Q4/Q5/W8 never had this problem because they ship this other assignment too --
// q4_volta_mma_gemm.cuh -- T on the 32-row A axis, N on the 8-row B axis, one warp per 8 output
// rows, dequantizing into *shared* memory once per K-pass and feeding mma.sync directly. NVFP4
// never got the equivalent. This is that kernel.
//
// Copied verbatim from q4_volta_mma_gemm.cuh: the schedule (4 warps, 32-row T tile, double-
// buffered shared staging, split-K via volta_mma_splits.h's shared CTA-budget logic), the
// pipeline shape (issue loads, run MMA on the *previous* tile, only then decode-and-store the
// new one -- fusing load+decode+store into one step left global latency exposed with nothing to
// hide it behind, per that file's own note), and the fp32-partial/atomicAdd/narrow structure for
// splits>1. What's new is entirely in prefetch()/commit(): NVFP4's decoder (from
// nvfp4_volta_qpn_gemm.cuh, not re-derived) and its swizzled scale plane.
//
// The one structural wrinkle: mma.sync.m8n8k4's B-fragment (the weight's own 32-row axis for the
// QPN kernel, but here the *A*-adjacent 8-row axis feeding volta_load_k) needs each warp's 8 rows'
// worth of decoded weight in *natural adjacent-k* half2 pairs -- exactly what Q4's magic-number
// decode already produces, because Q4's stored nibble order already is fragment order. NVFP4's
// e2m1 shift decode does not have that property (see nvfp4_volta_qpn_gemm.cuh's file header): one
// 32-bit word (8 nibbles, half a 16-k group) decodes to (i, i+4) pairs, not (i, i+1). The QPN
// kernel's fix was to permute the *activations* to match, because both operands funnel through
// the same mma and a shared permutation is free. Here activations are staged once per CTA and
// reused by all N rows in the warp's tile, so permuting them isn't free the same way -- instead
// the decoded weight is un-permuted back into adjacent-k order at store time, four cheap
// shift+mask ops (no byte_perm needed: half2.x/.y are already the low/high 16 bits of one 32-bit
// register), so x_sh's staging is untouched and identical to Q4's.
//
// Code-plane addressing turned out simpler than group semantics might suggest: NVFP4's code plane
// is plain row-major with no group-boundary padding (unlike the scale plane), so the byte offset
// for k=kbase is just kbase/2 regardless of where a 16-k group starts -- identical in form to
// Q4's crow + boff + (lane&3)*4, no per-group modulo needed, because kKStep=32 is already an
// exact multiple of the 16-k group size.

#include "ops/common/volta_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Nvfp4VoltaMmaSchedule {
    static constexpr int kWarps      = 4;
    static constexpr int kKStep      = 32; // exactly two 16-k NVFP4 groups per iteration
    static constexpr int kTTile      = 32;
    static constexpr int kXPad       = 8;
    static constexpr int kRowsPerCta = kWarps * 8;
    static constexpr int kThreads    = kWarps * 32;
};

// Un-permutes nvfp4_decode_e2m1_quad's (i, i+4) output into natural adjacent-k half2 pairs.
// half2.x/.y are bits [15:0]/[31:16] of the register, so this is pure shift+mask, no byte_perm.
__device__ __forceinline__ void nvfp4_mma_unshuffle(const half2 (&out)[4], half2 (&adj)[4]) {
    const auto* o           = reinterpret_cast<const std::uint32_t*>(out);
    auto* a                 = reinterpret_cast<std::uint32_t*>(adj);
    a[0] = (o[0] & 0x0000FFFFu) | (o[1] << 16);           // (k0, k1)
    a[1] = (o[2] & 0x0000FFFFu) | (o[3] << 16);           // (k2, k3)
    a[2] = (o[0] >> 16) | (o[1] & 0xFFFF0000u);           // (k4, k5)
    a[3] = (o[2] >> 16) | (o[3] & 0xFFFF0000u);           // (k6, k7)
}

template <bool kDirect>
__global__ __launch_bounds__(Nvfp4VoltaMmaSchedule::kThreads, 8) void nvfp4_volta_mma_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, float* __restrict__ partial,
    __nv_bfloat16* __restrict__ out, int out_ld, int n, int k, int t,
    float inverse_weight_divisor, int splits) {
    using S = Nvfp4VoltaMmaSchedule;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n0   = (static_cast<int>(blockIdx.x) * S::kWarps + warp) * 8;
    const int t0   = static_cast<int>(blockIdx.z) * S::kTTile;
    const int tcnt = min(S::kTTile, t - t0);

    __shared__ __align__(16) __half x_sh[2][S::kTTile][S::kKStep + S::kXPad];
    __shared__ __align__(16) __half w_sh[2][S::kWarps][8][S::kKStep + S::kXPad];

    const int chunk  = (k / splits) & ~(S::kKStep - 1);
    const int kstart = static_cast<int>(blockIdx.y) * chunk;
    const int kend   = (static_cast<int>(blockIdx.y) == splits - 1) ? k : kstart + chunk;
    if (kstart >= kend) { return; }

    const int scale_tiles = k / 64; // BlockScaleK16M128x4 tile stride, 4 groups per tile

    constexpr int kVecs = S::kKStep / 8;
    struct Carry {
        uint4 xraw;
        std::uint32_t word;
        std::uint8_t scale_byte;
        bool xactive;
        bool row_good;
    };
    Carry carry;

    const half2 rebias   = __float2half2_rn(16384.0f);
    const half2 divisor2 = __float2half2_rn(inverse_weight_divisor * 256.0f);

    auto prefetch = [&](int kbase, Carry& r) {
        const int idx = static_cast<int>(threadIdx.x);
        r.xactive     = idx < tcnt * kVecs;
        if (r.xactive) {
            const int row = idx / kVecs;
            const int v   = idx % kVecs;
            r.xraw        = *reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(t0 + row) * k + kbase + v * 8);
        }
        // Each of a warp's 8 output rows is served by 4 lanes; lane (row*4 + q) holds k-window
        // [kbase + q*8, kbase + q*8 + 8) of that row -- half of one 16-k group (q in {0,1}) or
        // the other half (q in {2,3}).
        const int r_row = lane >> 2;
        const int q     = lane & 3;
        const int row   = n0 + r_row;
        r.row_good      = row < n;
        r.word          = 0;
        r.scale_byte    = 0;
        if (r.row_good) {
            const std::uint8_t* crow = codes + static_cast<std::int64_t>(row) * (k / 2);
            r.word = *reinterpret_cast<const std::uint32_t*>(crow + (kbase + q * 8) / 2);

            const int group        = kbase / 16 + (q >> 1);
            const int m_tile        = row / 128;
            const int row_inner      = row - m_tile * 128;
            const int scale_tile     = group / 4;
            const int scale_lane     = group & 3;
            const std::int64_t off = static_cast<std::int64_t>(m_tile) * scale_tiles * 512 +
                                     static_cast<std::int64_t>(scale_tile) * 512 +
                                     (row_inner & 31) * 16 + (row_inner >> 5) * 4 + scale_lane;
            r.scale_byte = scales[off];
        }
    };

    auto commit = [&](const Carry& r, int buf) {
        if (r.xactive) {
            const int idx   = static_cast<int>(threadIdx.x);
            const auto* src = reinterpret_cast<const __nv_bfloat16*>(&r.xraw);
            __half tmp[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) { tmp[j] = __float2half(__bfloat162float(src[j])); }
            *reinterpret_cast<uint4*>(&x_sh[buf][idx / kVecs][(idx % kVecs) * 8]) =
                *reinterpret_cast<const uint4*>(tmp);
        }
        const int r_row = lane >> 2;
        const int q     = lane & 3;
        const half2 sc2 = __hmul2(nvfp4_decode_e4m3_scale(r.scale_byte), divisor2);
        half2 raw[4];
        nvfp4_decode_e2m1_quad(r.word, rebias, raw);
        half2 adj[4];
        nvfp4_mma_unshuffle(raw, adj);
#pragma unroll
        for (int j = 0; j < 4; ++j) { adj[j] = __hmul2(adj[j], sc2); }
        // q directly indexes this lane's 8-k window within the 32-wide kKStep tile (k offsets
        // 0,8,16,24), so the half-unit store offset is simply q*8.
        *reinterpret_cast<uint4*>(&w_sh[buf][warp][r_row][q * 8]) =
            *reinterpret_cast<const uint4*>(adj);
    };

    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    prefetch(kstart, carry);
    commit(carry, 0);
    int buf = 0;
    for (int kbase = kstart; kbase < kend; kbase += S::kKStep) {
        __syncthreads();
        const int nxt       = kbase + S::kKStep;
        const bool has_next = nxt < kend;
        if (has_next) { prefetch(nxt, carry); }

#pragma unroll
        for (int kk = 0; kk < S::kKStep; kk += 8) {
            half2 a[4], b[4];
            volta_load_qp(a, reinterpret_cast<const half2*>(&x_sh[buf][0][kk]),
                          (S::kKStep + S::kXPad) / 2);
            volta_load_k(b, reinterpret_cast<const half2*>(&w_sh[buf][warp][0][kk]),
                         (S::kKStep + S::kXPad) / 2);
            volta_mma_qk(d, a, b);
        }

        if (has_next) { commit(carry, buf ^ 1); }
        buf ^= 1;
    }

#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int row_t = volta_d_get_i(l);
        const int col_n = n0 + volta_d_get_j(l);
        if (row_t < tcnt && col_n < n) {
            if constexpr (kDirect) {
                out[static_cast<std::int64_t>(t0 + row_t) * out_ld + col_n] =
                    __float2bfloat16(d[l]);
            } else {
                atomicAdd(&partial[static_cast<std::int64_t>(t0 + row_t) * n + col_n], d[l]);
            }
        }
    }
}

__global__ void nvfp4_volta_mma_narrow_strided_kernel(const float* __restrict__ partial,
                                                       __nv_bfloat16* __restrict__ out, int n,
                                                       int t, int out_ld) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(n) * t) { return; }
    const int col = static_cast<int>(i / n);
    const int row = static_cast<int>(i % n);
    out[static_cast<std::int64_t>(col) * out_ld + row] = __float2bfloat16(partial[i]);
}

#endif // sm_70

} // namespace ninfer::ops::detail
