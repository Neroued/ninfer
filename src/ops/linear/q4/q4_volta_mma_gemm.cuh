#pragma once

// Fused-dequant Q4 x BF16 GEMM on Volta tensor cores (mma.sync.m8n8k4), sm_70 only.
//
// Both pre-existing routes are structurally wrong for the T values concurrency produces
// (with MTP3, T = 4C, so C1/2/4/8 sit at T = 4/8/16/32):
//
//   - the SIMT route (q4_simt_r8_c8) is ALU-bound at roughly 22 ops per weight byte and
//     re-reads the whole weight matrix once per 8 output columns;
//   - the CUTLASS Sm70 routes elsewhere in this port do use tensor cores, but dequantise
//     the entire weight into an FP16 scratch buffer in *global* memory on every call, so
//     they move several times the bytes the problem actually requires and their cost is
//     nearly independent of T.
//
// This kernel dequantises Q4 into *shared* memory and feeds mma.sync.m8n8k4 directly, so
// the weight is read once per K pass at Q4 density and the multiply-accumulate runs on the
// tensor cores. Measured on a V100-SXM2-32GB at N=4096 K=5120 against q4_simt_r8_c8, with
// L2 relative error 0.223% against an fp32 host reference (the CUTLASS Sm70 paths in this
// port document 0.17% for the same reason -- fp16 operands, fp32 accumulate):
//
//     T:        8       16       32       48       64
//     SIMT:  79.9us  137.2us  259.1us  348.2us  456.7us
//     this:  83.5us   87.3us  102.1us  177.2us  188.4us
//
// so it reaches parity around T=8 and holds 2-2.5x above T=16.
//
// Fragment mapping. volta_mma_qk computes D[32x8 f32] += A[32x8 f16] @ B[8x8 f16]^T with 8
// real k-elements per call, so T maps to the 32-row A axis and N to the 8-row B axis, and
// one warp owns 8 output rows. The fragment addressing itself lives in ops/common/volta_mma.cuh
// and is not re-derived here.
//
// Split-K is load-bearing, not incidental: each CTA otherwise walks all of K serially, and at
// N=4096 K=5120 T=32 the measured cost is 250us at 1 split against 125us at 4. Increasing CTA
// count through the N dimension instead does not substitute for it (best workspace-free
// configuration measured 253.7us, i.e. parity with SIMT). Partials therefore accumulate into an
// fp32 workspace and a second pass narrows to BF16.

#include "ops/common/volta_mma.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

struct Q4VoltaMmaSchedule {
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
// `kDirect` is the one-split case, which after the split-K retune is what the widest and most
// frequent shape (gate/up, n=34816) actually runs. With a single split each CTA owns its output
// tile outright, so the whole fp32 apparatus -- workspace, zeroing memset, atomicAdd, narrowing
// pass -- is pure overhead against a plain BF16 store. Same structure the W8 kernel is built on.
template <bool kDirect>
__global__ __launch_bounds__(Q4VoltaMmaSchedule::kThreads, 8) void q4_volta_mma_gemm_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, float* __restrict__ partial,
    __nv_bfloat16* __restrict__ out, int out_ld, int n, int k, int t, int padded_groups,
    int splits) {
    using S = Q4VoltaMmaSchedule;
    constexpr int kGroupK = Q4RowSplitStorage::kGroupK;
    constexpr int kCodeB  = Q4RowSplitStorage::kCodeBytesPerGroup;

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

    // The pipeline is split into a global-load half and a decode-and-store half, held apart by
    // this register carrier, because fusing them is what the first version got wrong: it staged
    // an entire tile -- load, decode, store -- and only then issued MMA on the previous one, so
    // the shared store waited on the global load with no independent work in between. ncu put
    // 6.58 of 15.8 warp-cycles-per-issued-instruction in long_scoreboard, i.e. global latency
    // nothing was hiding. Issuing the loads, running the MMA, and only then consuming the
    // registers lets the whole MMA loop sit inside that latency.
    //
    // tcnt * kVecs is at most 32 * 4 = 128 = kThreads, so the activation staging is exactly one
    // uint4 per thread and the carrier stays small enough not to cost occupancy.
    constexpr int kVecs = S::kKStep / 8;
    struct Carry {
        uint4 xraw;
        std::uint32_t p4;
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
        r.sc           = __ushort_as_half(0);
        if (row < n) {
            const std::uint8_t* crow =
                codes + static_cast<std::int64_t>(row) * padded_groups * kCodeB;
            r.p4 = *reinterpret_cast<const std::uint32_t*>(
                crow + static_cast<std::int64_t>(g) * kCodeB + boff + (lane & 3) * 4);
            r.sc = __ushort_as_half(reinterpret_cast<const std::uint16_t*>(
                scales + static_cast<std::int64_t>(row) * padded_groups *
                             Q4RowSplitStorage::kScaleBytesPerGroup)[g]);
        }
    };

    // Magic-number decode kept entirely in fp16: 0x6400|n is exactly 1024+n as fp16, so one
    // __hsub2 against 1032.0 recovers the code and one __hmul2 applies the scale. The codes are
    // two's-complement, so each nibble is first biased by XOR 8 -- (n ^ 8) - 8 gives n for n < 8
    // and n - 16 above it. Same identity Q4SimtDecodeAtom uses, but without its trip out to fp32
    // and back, which measured 1.2-1.3x on this kernel across T. kKStep/2 = 16 code bytes per row
    // over 8 rows is exactly 32 lanes x 4 bytes, so every lane emits one aligned 16-byte store.
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
        const int b0           = (lane & 3) * 4;
        const std::uint32_t p4 = r.p4 ^ 0x88888888u;
        const half2 sc2        = __half2half2(r.sc);
        const half2 bias       = __half2half2(__ushort_as_half(0x6408)); // 1032.0
        half2 decoded[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t byte = (p4 >> (8 * j)) & 0xffu;
            std::uint32_t bits = ((byte & 0x0fu) | ((byte & 0xf0u) << 12)) | 0x64006400u;
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
        const int nxt        = kbase + S::kKStep;
        const bool has_next  = nxt < kend;
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

__global__ void q4_volta_mma_narrow_kernel(const float* __restrict__ partial,
                                           __nv_bfloat16* __restrict__ out, std::int64_t count) {
    const std::int64_t i =
        static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = __float2bfloat16(partial[i]); }
}

#endif // sm_70

} // namespace ninfer::ops::detail
