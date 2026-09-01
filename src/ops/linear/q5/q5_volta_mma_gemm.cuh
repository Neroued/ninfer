#pragma once

// Fused-dequant Q5 x BF16 GEMM on Volta tensor cores (mma.sync.m8n8k4), sm_70 only, with an
// optional in-place residual accumulate for the linear_add form.
//
// Companion to q4_volta_mma_gemm.cuh; see that file for why neither the SIMT nor the CUTLASS
// Sm70 route suits the token range concurrency produces. The Q5 case matters more in aggregate:
// mlp/down (64 layers, k=17408) plus gdn/output and attn/output (64 layer-instances, k=6144) all
// sit on linear_add/q5's CUTLASS route, whose cost is a nearly T-independent ~1250us / ~459us
// dominated by materialising the dequantised weight into global FP16 on every call.
//
// Measured on a V100-SXM2-32GB against the production linear_add/q5 route, back to back:
//
//   k=17408 [5120,17408]   T=12  634 -> 433us    T=16  949 -> 456us    T=32 1253 -> 552us
//   k=6144  [5120,6144]    T=12  150 -> 148us    T=16  406 -> 154us    T=32  459 -> 184us
//
// L2 relative error against an fp32 host reference is 0.223%, matching the Q4 kernel.
//
// Q5 codec. Codes are two's-complement 5-bit, split across a 32-byte low-nibble plane and an
// 8-byte high-bit plane per 64-weight group. The decode stays entirely in fp16: 0x6400|c is
// exactly 1024+c, the 5th bit lands at position 4 (low half) and 20 (high half), and the whole
// high byte is pre-XORed with 0xff so one __hsub2 against 1040.0 recovers the signed code. Same
// identity Q5SimtDecodeAtom uses, without its trip out to fp32.
//
// Split-K count is K-dependent and measured, not derived: at T=32, k=5120 wants 4 splits, k=6144
// wants 8 (196.9/184.3/193.2/223.5us at 4/8/16/32) and k=17408 wants 32 (1045/813/695/552/556us
// at 4/8/16/32/48). Chunk size is not the invariant -- see q5_volta_mma_splits().

#include "ops/common/volta_mma.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Q5VoltaMmaSchedule {
    static constexpr int kWarps  = 4;  // warps per CTA; each owns 8 output rows
    static constexpr int kKStep  = 32; // K elements staged per iteration (double-buffered)
    static constexpr int kTTile  = 32; // rows of the mma A operand
    static constexpr int kXPad   = 8;  // shared row padding. Unpadded, every lane of the
                                       // A-fragment load hits bank 0 (32-way conflict). Padded,
                                       // 8 distinct banks remain -- which is the hardware floor
                                       // here, not a residual conflict: the load is 32 lanes x
                                       // 16B = 512B against 128B/cycle of shared bandwidth, so
                                       // 4 cycles is optimal. An XOR swizzle would not improve it.
    static constexpr int kRowsPerCta = kWarps * 8;
    static constexpr int kThreads    = kWarps * 32;
};

// Partials accumulate in fp32 and are narrowed afterwards, so split-K contributions from
// different blockIdx.y values can be combined without losing the low bits of the sum.
//
// `kDirect` is the one-split case: each CTA then owns its output tile outright, so the fp32
// workspace, its memset, the atomics and the narrowing pass are all pure overhead against a plain
// BF16 store. `kAddResidual` folds linear_add's beta=1 epilogue into that store, reading `out` as
// C and writing it back as D exactly as the narrowing kernel would have.
template <bool kDirect, bool kAddResidual>
__global__ __launch_bounds__(Q5VoltaMmaSchedule::kThreads, 8) void q5_volta_mma_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ high,
    const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, float* __restrict__ partial,
    __nv_bfloat16* __restrict__ out, int out_ld, int n, int k, int t, int padded_groups,
    int splits) {
    using S = Q5VoltaMmaSchedule;
    constexpr int kGroupK = Q5RowSplitStorage::kGroupK;
    constexpr int kCodeB  = Q5RowSplitStorage::kCodeBytesPerGroup;
    constexpr int kHighB  = Q5RowSplitStorage::kHighBytesPerGroup;

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int n0   = (static_cast<int>(blockIdx.x) * S::kWarps + warp) * 8;
    const int t0   = static_cast<int>(blockIdx.z) * S::kTTile;
    const int tcnt = min(S::kTTile, t - t0);

    __shared__ __align__(16) __half x_sh[2][S::kTTile][S::kKStep + S::kXPad];
    __shared__ __align__(16) __half w_sh[2][S::kWarps][8][S::kKStep + S::kXPad];

    // Each split owns a kKStep-aligned run of K; the last one absorbs the remainder.
    const int chunk  = (k / splits) & ~(S::kKStep - 1);
    const int kstart = static_cast<int>(blockIdx.y) * chunk;
    const int kend   = (static_cast<int>(blockIdx.y) == splits - 1) ? k : kstart + chunk;
    if (kstart >= kend) { return; }

    // Split into a global-load half and a decode-and-store half held apart by a register carrier,
    // so the MMA loop runs inside the global latency instead of after it. See the Q4 sibling for
    // the ncu evidence that made this the right change (long_scoreboard 6.58 -> 2.22 of 15.8
    // warp-cycles per issued instruction, at no cost in registers or occupancy).
    constexpr int kVecs = S::kKStep / 8;
    struct Carry {
        uint4 xraw;
        std::uint32_t p4;
        std::uint32_t hb;
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
        // 8 rows x kKStep weights, fetched once for the warp. Doing this per lane instead would
        // repeat the work 4x, because volta_load_k's I-major-mirrored B fragment gives only 8
        // distinct rows across the 32 lanes.
        const int g    = kbase / kGroupK;
        const int boff = (kbase % kGroupK) >> 1;
        const int row  = n0 + (lane >> 2);
        r.p4           = 0;
        r.hb           = 0xffu;
        r.sc           = __ushort_as_half(0);
        if (row < n) {
            const std::uint8_t* crow =
                codes + static_cast<std::int64_t>(row) * padded_groups * kCodeB;
            r.p4 = *reinterpret_cast<const std::uint32_t*>(
                crow + static_cast<std::int64_t>(g) * kCodeB + boff + (lane & 3) * 4);
            const std::uint8_t* hrow =
                high + static_cast<std::int64_t>(row) * padded_groups * kHighB;
            r.hb = static_cast<std::uint32_t>(
                       hrow[static_cast<std::int64_t>(g) * kHighB + (boff >> 2) + (lane & 3)]) ^
                   0xffu;
            r.sc = __ushort_as_half(reinterpret_cast<const std::uint16_t*>(
                scales + static_cast<std::int64_t>(row) * padded_groups *
                             Q5RowSplitStorage::kScaleBytesPerGroup)[g]);
        }
    };

    // Magic-number decode kept entirely in fp16. 0x6400|c is exactly 1024+c as fp16; the 5th code
    // bit sits at bit 4 of the low half and bit 20 of the high half. The high byte is pre-XORed
    // with 0xff so that subtracting 1040.0 yields the signed two's-complement code directly. One
    // high byte covers 8 weights = 4 low bytes = exactly this lane's slice.
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
        const int r_row  = lane >> 2;
        const int b0     = (lane & 3) * 4;
        const half2 sc2  = __half2half2(r.sc);
        const half2 bias = __half2half2(__ushort_as_half(0x6410)); // 1040.0
        half2 decoded[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t lb = (r.p4 >> (8 * j)) & 0xffu;
            std::uint32_t bits = ((lb & 0x0fu) | ((lb & 0xf0u) << 12)) | 0x64006400u;
            bits |= (((r.hb >> (2 * j)) & 1u) << 4) | (((r.hb >> (2 * j + 1)) & 1u) << 20);
            decoded[j] = __hmul2(__hsub2(*reinterpret_cast<half2*>(&bits), bias), sc2);
        }
        *reinterpret_cast<uint4*>(&w_sh[buf][warp][r_row][2 * b0]) =
            *reinterpret_cast<const uint4*>(decoded);
    };

    float d[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    prefetch(kstart, carry);
    commit(carry, 0);
    int buf = 0;
    // One barrier per iteration is still enough. Iteration i reads buf and writes buf^1, and the
    // only conflict -- iteration i-1's reads of the buffer iteration i is about to write -- is
    // separated by the barrier at the top, because that whole body precedes it.
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
                const std::int64_t o =
                    static_cast<std::int64_t>(t0 + row_t) * out_ld + col_n;
                float v = d[l];
                if constexpr (kAddResidual) { v += __bfloat162float(out[o]); }
                out[o] = __float2bfloat16(v);
            } else {
                atomicAdd(&partial[static_cast<std::int64_t>(t0 + row_t) * n + col_n], d[l]);
            }
        }
    }
}

// `out` is the linear_add residual: read as C and written back as D, matching the beta=1
// epilogue the CUTLASS route uses. With kAddResidual false this is a plain narrowing store.
template <bool kAddResidual>
__global__ void q5_volta_mma_narrow_kernel(const float* __restrict__ partial,
                                           __nv_bfloat16* __restrict__ out, int n, int t,
                                           int out_ld) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<std::int64_t>(n) * t) { return; }
    const int col = static_cast<int>(i / n);
    const int row = static_cast<int>(i % n);
    const std::int64_t o = static_cast<std::int64_t>(col) * out_ld + row;
    float v = partial[i];
    if constexpr (kAddResidual) { v += __bfloat162float(out[o]); }
    out[o] = __float2bfloat16(v);
}

#endif // sm_70

} // namespace ninfer::ops::detail
