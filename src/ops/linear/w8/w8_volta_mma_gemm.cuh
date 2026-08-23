#pragma once

// Fused-dequant W8 x BF16 GEMM on Volta tensor cores (mma.sync.m8n8k4), sm_70 only.
//
// Same design as q4_volta_mma_gemm.cuh -- dequantise into *shared* memory and feed the tensor
// cores directly, so the weight is read once per K pass at its stored density -- applied to the
// one shape that needed it most. The 27B output head is W8 [248320,5120], and on Volta every
// tensor-core W8 schedule traps below sm_80, so it fell back to sliced r8_c8 SIMT. That route
// re-reads the entire 1.31 GB weight once per 8 output columns, which makes its cost exactly
// linear in T: 7.65 / 14.36 / 21.33 / 28.49 ms at T=8/16/24/32, i.e. 3.9 GB moved at T=24 where
// 1.31 GB is necessary. At 14% of a C8 round it was the largest unfused item left.
//
// Two things differ from the Q4/Q5 siblings, both simplifications:
//
//   - No split-K, and therefore no fp32 workspace, no zeroing memset, no narrowing pass and no
//     atomics. Split-K exists in those kernels because N=4096-34816 does not by itself supply
//     enough CTAs; N=248320 supplies 7760, twelve times what the V100 holds resident, so the
//     shared CTA budget (ops/common/volta_mma_splits.h) picks 1 split at every T regardless.
//     Each CTA therefore owns its output tile outright and stores BF16 straight to `out`.
//   - kGroupK is 32 for W8 against 64 for Q4, which is exactly kKStep, so one staging step
//     consumes exactly one group per row and the scale is loop-invariant within the step.
//
// Decode is the 8-bit form of the sibling kernels' magic-number identity, kept entirely in fp16:
// 0x6400|u is exactly 1024+u as fp16 for u in [0,1023], so biasing a code byte by XOR 0x80 into
// u = b+128 makes 0x6400|u equal 1152+b, and one __hsub2 against 1152.0 recovers the signed code
// while one __hmul2 applies the scale. Two codes per half2, four half2 per lane per step.

#include "ops/common/volta_mma.cuh"
#include "ops/linear/w8/w8_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct W8VoltaMmaSchedule {
    static constexpr int kWarps = 4;  // warps per CTA; each owns 8 output rows
    static constexpr int kKStep = 32; // K elements staged per iteration (double-buffered)
    static constexpr int kTTile = 32; // rows of the mma A operand
    static constexpr int kXPad  = 8;  // shared row padding; see the Q4 sibling for why 8 is the
                                      // hardware floor here rather than a residual conflict.
    static constexpr int kRowsPerCta = kWarps * 8;
    static constexpr int kThreads    = kWarps * 32;
};

__global__ __launch_bounds__(W8VoltaMmaSchedule::kThreads, 8) void w8_volta_mma_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int n, int k, int t,
    int padded_groups, int out_ld) {
    using S = W8VoltaMmaSchedule;
    constexpr int kGroupK = W8RowSplitStorage::kGroupK;
    constexpr int kCodeB  = W8RowSplitStorage::kCodeBytesPerGroup;
    static_assert(kGroupK == S::kKStep, "one staging step must consume exactly one W8 group");

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n0   = (static_cast<int>(blockIdx.x) * S::kWarps + warp) * 8;
    const int t0   = static_cast<int>(blockIdx.y) * S::kTTile;
    const int tcnt = min(S::kTTile, t - t0);

    __shared__ __align__(16) __half x_sh[2][S::kTTile][S::kKStep + S::kXPad];
    __shared__ __align__(16) __half w_sh[2][S::kWarps][8][S::kKStep + S::kXPad];

    // Split into a global-load half and a decode-and-store half held apart by a register carrier,
    // so the MMA loop runs inside the global latency instead of after it. See the Q4 sibling for
    // the ncu evidence behind this shape.
    constexpr int kVecs = S::kKStep / 8;
    struct Carry {
        uint4 xraw;
        uint2 p8;
        __half sc;
        bool xactive;
    };
    Carry carry;

    auto prefetch = [&](int kbase, Carry& r) {
        const int idx = static_cast<int>(threadIdx.x);
        r.xactive     = idx < tcnt * kVecs;
        if (r.xactive) {
            const int row = idx / kVecs;
            const int v   = idx % kVecs;
            r.xraw        = *reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(t0 + row) * k + kbase + v * 8);
        }
        // 8 rows x kKStep weights, fetched once for the warp -- volta_load_k's I-major-mirrored B
        // fragment gives only 8 distinct rows across the 32 lanes, so doing this per lane would
        // repeat the work 4x. kKStep = 32 code bytes per row over 8 rows is exactly 32 lanes x 8
        // bytes, so every lane makes one aligned 8-byte load and one aligned 16-byte store.
        const int g   = kbase / kGroupK;
        const int row = n0 + (lane >> 2);
        r.p8          = make_uint2(0, 0);
        r.sc          = __ushort_as_half(0);
        if (row < n) {
            const std::uint8_t* crow =
                codes + static_cast<std::int64_t>(row) * padded_groups * kCodeB;
            r.p8 = *reinterpret_cast<const uint2*>(
                crow + static_cast<std::int64_t>(g) * kCodeB + (lane & 3) * 8);
            r.sc = __ushort_as_half(reinterpret_cast<const std::uint16_t*>(
                scales + static_cast<std::int64_t>(row) * padded_groups *
                             W8RowSplitStorage::kScaleBytesPerGroup)[g]);
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
        const int r_row        = lane >> 2;
        const int b0           = (lane & 3) * 8;
        const half2 sc2        = __half2half2(r.sc);
        const half2 bias       = __half2half2(__ushort_as_half(0x6480)); // 1152.0
        const std::uint32_t w0 = r.p8.x ^ 0x80808080u;
        const std::uint32_t w1 = r.p8.y ^ 0x80808080u;
        half2 decoded[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t src = (j < 2) ? w0 : w1;
            const int shift         = (j & 1) * 16;
            std::uint32_t bits =
                (((src >> shift) & 0xffu) | (((src >> shift) & 0xff00u) << 8)) | 0x64006400u;
            decoded[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
        *reinterpret_cast<uint4*>(&w_sh[buf][warp][r_row][b0]) =
            *reinterpret_cast<const uint4*>(decoded);
    };

    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    prefetch(0, carry);
    commit(carry, 0);
    int buf = 0;
    // One barrier per iteration is still enough. Iteration i reads buf and writes buf^1, and the
    // only conflict -- iteration i-1's reads of the buffer iteration i is about to write -- is
    // separated by the barrier at the top, because that whole body precedes it.
    for (int kbase = 0; kbase < k; kbase += S::kKStep) {
        __syncthreads();
        const int nxt       = kbase + S::kKStep;
        const bool has_next = nxt < k;
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

    // One split, so this CTA owns the whole dot product for its tile: no atomics, no fp32
    // workspace, no narrowing pass. Straight to BF16 through out's real column stride.
#pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int row_t = volta_d_get_i(l);
        const int col_n = n0 + volta_d_get_j(l);
        if (row_t < tcnt && col_n < n) {
            out[static_cast<std::int64_t>(t0 + row_t) * out_ld + col_n] = __float2bfloat16(d[l]);
        }
    }
}

#endif // sm_70

} // namespace ninfer::ops::detail
