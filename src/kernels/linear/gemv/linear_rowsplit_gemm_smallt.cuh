#pragma once

// Warp-per-row small-T row-split low-bit GEMM: out[N,T] = W[N,K] . x[K,T].
//
// This is the universal low-bit path for every T the tuned decode GEMVs and the
// LargeT tensor-core GEMM do not own: SmallT (2..16) for Q4/Q5/Q6, all T for
// W8G32, generic-shape T==1, and the k%8!=0 LargeT fallback. The design target
// is the DRAM roofline: stream the weight payload once per kTt-column tile at
// copy-ceiling bandwidth, so the cost of T<=kTt is nearly flat versus T==1 (the
// property MTP verify needs).
//
//   - One warp owns one output row. blockIdx.y selects a tile of kTt activation
//     columns; for T <= kTt the weights are streamed exactly once.
//   - K is processed in 1024-value slabs staged through shared memory with a
//     cp.async double buffer (same structure as the tuned T==1 Q5 core): all
//     weight-plane reads are coalesced 128-bit loads, so DRAM latency is hidden
//     without relying on occupancy alone. Deeper pipelines and launch-bounds
//     occupancy caps were both measured slower (spills / no gain, see report).
//   - Consume is phase-interleaved: in phase c (0..3), lane L owns the 8
//     consecutive K-values [256c + 8L, 256c + 8L + 8). The warp's 32 x loads per
//     (column, phase) are one 16-byte uint4 per lane covering 512 *consecutive*
//     bytes, i.e. minimal L1 wavefronts (a per-lane-contiguous layout would
//     stride lanes 64B apart and burn 4x the L1 throughput on the same bytes).
//     The weight planes index out conflict-free or broadcast under the same
//     ownership.
//   - Dequant uses the fp16 mantissa trick: nibbles are OR-ed into the mantissa
//     of 1024.0f16 (plus high bits for Q5/Q6 at mantissa bits 4..5), the sign
//     fixup is a constant xor folded into the packed word, and one hsub2 yields
//     two exact signed values; the group scale is applied per 8-value chunk.
//     fp32 FMA into per-column accumulators; warp-shuffle reduction per column.
//   - kTt is 4 or 8 only. Larger column tiles blow past the register budget
//     (kTt=16 needs ~98 regs -> 2 blocks/SM -> latency-bound at ~20% of DRAM),
//     so T > 8 re-streams the weights once per 8-column tile instead; the mma
//     tensor-core GEMM takes over where that stops winning.
//
// Correctness for arbitrary shapes: full 1024-value slabs require k % 8 == 0
// (16-byte aligned x columns) and cover [0, k/1024*1024); every remaining group
// (tail of odd k, or all groups when k % 8 != 0) uses the scalar per-pair path
// reading global memory directly, masked at the k boundary. Weights in the
// padded region [k, padded_k) are never used.

#include "kernels/linear/codec/linear_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include <cstdint>
#include <type_traits>

namespace qus::kernels::detail {

// Per-codec slab traits + dequant. A slab is 1024 K-values:
//   nibble/code bytes : 32 B per group  -> kNibU4 uint4 per slab
//   high-plane bytes  : 0/8/16 B per group -> kHighU4 uint4 per slab
//   scales            : 2 B per group -> kScaleU32 4-byte words per slab
// (Scales are staged with 4-byte cp.async because a row's scale plane is only
// guaranteed 4-byte aligned for generic kg.)
//
// dequant_chunk(s_nib, s_hi, s_sc, c, lane, w): dequantize the lane's 8 values
// of phase c into w[0..7], scale applied. All shared reads are conflict-free or
// broadcast under the phase-interleaved ownership.

struct Q4Smallt {
    using Codec = Q4Codec;
    static constexpr int kNibU4  = 32;
    static constexpr int kHighU4 = 0;
    static constexpr int kScaleU32 = 8;
    static constexpr int kHighBytesPerGroup = 0;

    // Nibble j of the code word is value j; (v ^ 8) - 8 sign extension becomes
    // xor 0x8 per nibble + subtract (1024 + 8) in the fp16 domain.
    __device__ static __forceinline__ void
    dequant_chunk(const uint4* s_nib, const uint4* /*s_hi*/, const std::uint32_t* s_sc, int c,
                  int lane, float (&w)[8]) {
        const std::uint32_t word =
            reinterpret_cast<const std::uint32_t*>(s_nib)[c * 32 + lane] ^ 0x88888888u;
        const float scale = __half2float(__ushort_as_half(
            reinterpret_cast<const std::uint16_t*>(s_sc)[c * 4 + (lane >> 3)]));
        const __half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032.0
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            const __half2 h    = __hsub2(*reinterpret_cast<const __half2*>(&bits), bias);
            const float2  f    = __half22float2(h);
            w[p]     = f.x * scale;
            w[p + 4] = f.y * scale;
        }
    }
};

struct Q5Smallt {
    using Codec = Q5Codec;
    static constexpr int kNibU4  = 32;
    static constexpr int kHighU4 = 8;
    static constexpr int kScaleU32 = 8;
    static constexpr int kHighBytesPerGroup = 8;

    // u = lo | hi<<4, signed s = (u ^ 16) - 16 = lo + 16*(hi^1) - 16. Build
    // 1024 + lo + 16*(hi^1) in fp16 (hi^1 at mantissa bit 4), subtract 1040.
    __device__ static __forceinline__ void
    dequant_chunk(const uint4* s_nib, const uint4* s_hi, const std::uint32_t* s_sc, int c,
                  int lane, float (&w)[8]) {
        const std::uint32_t word =
            reinterpret_cast<const std::uint32_t*>(s_nib)[c * 32 + lane];
        const std::uint32_t hc =
            reinterpret_cast<const std::uint8_t*>(s_hi)[c * 32 + lane] ^ 0xffu;
        const float scale = __half2float(__ushort_as_half(
            reinterpret_cast<const std::uint16_t*>(s_sc)[c * 4 + (lane >> 3)]));
        const __half2 bias = __half2half2(__ushort_as_half(0x6410)); // 1040.0
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&bits), bias);
            const float2  f = __half22float2(h);
            w[p]     = f.x * scale;
            w[p + 4] = f.y * scale;
        }
    }
};

struct Q6Smallt {
    using Codec = Q6Codec;
    static constexpr int kNibU4  = 32;
    static constexpr int kHighU4 = 16;
    static constexpr int kScaleU32 = 8;
    static constexpr int kHighBytesPerGroup = 16;

    // u = lo | hi2<<4, signed s = (u ^ 32) - 32 = lo + 16*(hi2 ^ 2) - 32: flip
    // bit 1 of every 2-bit field (xor 0xaaaa), place at mantissa bits 4..5,
    // subtract 1056.
    __device__ static __forceinline__ void
    dequant_chunk(const uint4* s_nib, const uint4* s_hi, const std::uint32_t* s_sc, int c,
                  int lane, float (&w)[8]) {
        const std::uint32_t word =
            reinterpret_cast<const std::uint32_t*>(s_nib)[c * 32 + lane];
        const std::uint32_t hw =
            reinterpret_cast<const std::uint16_t*>(s_hi)[c * 32 + lane] ^ 0xaaaau;
        const float scale = __half2float(__ushort_as_half(
            reinterpret_cast<const std::uint16_t*>(s_sc)[c * 4 + (lane >> 3)]));
        const __half2 bias = __half2half2(__ushort_as_half(0x6420)); // 1056.0
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hw >> (2 * p)) & 3u) << 4) | (((hw >> (2 * p + 8)) & 3u) << 20);
            const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&bits), bias);
            const float2  f = __half22float2(h);
            w[p]     = f.x * scale;
            w[p + 4] = f.y * scale;
        }
    }
};

struct W8Smallt {
    using Codec = W8G32Codec;
    static constexpr int kNibU4  = 64;
    static constexpr int kHighU4 = 0;
    static constexpr int kScaleU32 = 16;
    static constexpr int kHighBytesPerGroup = 0;

    // Byte j is signed value j. The lane's 8 values sit inside one 32-value
    // group (offset 8L % 32), so a single scale applies.
    __device__ static __forceinline__ void
    dequant_chunk(const uint4* s_nib, const uint4* /*s_hi*/, const std::uint32_t* s_sc, int c,
                  int lane, float (&w)[8]) {
        const uint2 words = reinterpret_cast<const uint2*>(s_nib)[c * 32 + lane];
        const float scale = __half2float(__ushort_as_half(
            reinterpret_cast<const std::uint16_t*>(s_sc)[c * 8 + (lane >> 2)]));
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const std::uint32_t word = (&words.x)[j];
            w[4 * j + 0] = static_cast<float>(static_cast<std::int8_t>(word & 0xffu)) * scale;
            w[4 * j + 1] =
                static_cast<float>(static_cast<std::int8_t>((word >> 8) & 0xffu)) * scale;
            w[4 * j + 2] =
                static_cast<float>(static_cast<std::int8_t>((word >> 16) & 0xffu)) * scale;
            w[4 * j + 3] =
                static_cast<float>(static_cast<std::int8_t>((word >> 24) & 0xffu)) * scale;
        }
    }
};

template <class SC>
__device__ __forceinline__ void
smallt_issue_slab(uint4* __restrict__ s_nib, uint4* __restrict__ s_hi,
                  std::uint32_t* __restrict__ s_sc, const std::uint8_t* __restrict__ code_row,
                  const std::uint8_t* __restrict__ high_row,
                  const std::uint8_t* __restrict__ scale_row, int slab, int lane) {
    static_assert(SC::kNibU4 % 32 == 0, "nibble plane must be a whole number of warp copies");
#pragma unroll
    for (int j = 0; j < SC::kNibU4 / 32; ++j) {
        const int i = j * 32 + lane;
        __pipeline_memcpy_async(&s_nib[i],
                                code_row + static_cast<std::int64_t>(slab) * (SC::kNibU4 * 16) +
                                    i * 16,
                                16);
    }
    if constexpr (SC::kHighU4 > 0) {
        if (lane < SC::kHighU4) {
            __pipeline_memcpy_async(&s_hi[lane],
                                    high_row +
                                        static_cast<std::int64_t>(slab) * (SC::kHighU4 * 16) +
                                        lane * 16,
                                    16);
        }
    }
    if (lane < SC::kScaleU32) {
        __pipeline_memcpy_async(&s_sc[lane],
                                scale_row +
                                    static_cast<std::int64_t>(slab) * (SC::kScaleU32 * 4) +
                                    lane * 4,
                                4);
    }
    __pipeline_commit();
}

// x0 points at the column tile base (x + col0*k); xslab is the slab's first
// K-value. Requires k % 8 == 0 and 16-byte aligned x.
template <class SC, int kTt>
__device__ __forceinline__ void
smallt_consume_slab(const __nv_bfloat16* __restrict__ x0, std::int64_t xslab, std::int32_t k,
                    int ncols, const uint4* __restrict__ s_nib, const uint4* __restrict__ s_hi,
                    const std::uint32_t* __restrict__ s_sc, int lane, float (&acc)[kTt]) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        float w[8];
        SC::dequant_chunk(s_nib, s_hi, s_sc, c, lane, w);
        const std::int64_t xoff = xslab + c * 256 + lane * 8;
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            if (tt < ncols) {
                const uint4 xv = *reinterpret_cast<const uint4*>(
                    x0 + static_cast<std::int64_t>(tt) * k + xoff);
                const float2 f0 =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.x));
                const float2 f1 =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.y));
                const float2 f2 =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.z));
                const float2 f3 =
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.w));
                acc[tt] = fmaf(w[0], f0.x, acc[tt]);
                acc[tt] = fmaf(w[1], f0.y, acc[tt]);
                acc[tt] = fmaf(w[2], f1.x, acc[tt]);
                acc[tt] = fmaf(w[3], f1.y, acc[tt]);
                acc[tt] = fmaf(w[4], f2.x, acc[tt]);
                acc[tt] = fmaf(w[5], f2.y, acc[tt]);
                acc[tt] = fmaf(w[6], f3.x, acc[tt]);
                acc[tt] = fmaf(w[7], f3.y, acc[tt]);
            }
        }
    }
}

template <class SC, int kFullSlabs, int kStride>
__launch_bounds__(64, 16) __global__ void linear_rowsplit_gemm_smallt_kernel_direct_split2_q5_t4(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t n, std::int32_t k, std::int32_t t,
    std::int32_t padded_k, std::int32_t full_slabs) {
    static_assert(std::is_same_v<SC, Q5Smallt>, "direct split2 small-T kernel is Q5-only");
    static_assert(kFullSlabs > 0 && kStride > 0, "direct split2 requires exact positive shape");
    (void)full_slabs;
    (void)k;
    (void)t;

    __shared__ float s_part[2][4];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int part = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x);
    if (row >= n) { return; }

    const int           kg_padded = padded_k / Q5Codec::kGroupK;
    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row =
        high + static_cast<std::int64_t>(row) * kg_padded * Q5Smallt::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;

    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (int s = 0; s < kFullSlabs; ++s) {
#pragma unroll
        for (int local = 0; local < 2; ++local) {
            const int chunk = part * 2 + local;
            const std::uint8_t* code_phase =
                code_row + static_cast<std::int64_t>(s) * 512 + chunk * 128 + lane * 4;
            const std::uint8_t* high_phase =
                high_row + static_cast<std::int64_t>(s) * 128 + chunk * 32 + lane;
            const int group_in_slab = chunk * 4 + (lane >> 3);
            std::uint32_t scale_bits = 0;
            if ((lane & 7) == 0) {
                scale_bits = *reinterpret_cast<const std::uint16_t*>(
                    scale_row + (static_cast<std::int64_t>(s) * 16 + group_in_slab) * 2);
            }
            scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~7);

            const std::uint32_t word  = *reinterpret_cast<const std::uint32_t*>(code_phase);
            const std::uint32_t hc    = static_cast<std::uint32_t>(*high_phase) ^ 0xffu;
            const float         scale = __half2float(__ushort_as_half(scale_bits));
            const __half2       bias  = __half2half2(__ushort_as_half(0x6410)); // 1040.0
            float               w[8];
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
                bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
                const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&bits), bias);
                const float2  f = __half22float2(h);
                w[p]            = f.x * scale;
                w[p + 4]        = f.y * scale;
            }

            const std::int64_t xoff =
                static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8;
            const uint4 xv0 = *reinterpret_cast<const uint4*>(x + xoff);
            const uint4 xv1 = *reinterpret_cast<const uint4*>(x + kStride + xoff);
            const uint4 xv2 = *reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(2) * kStride + xoff);
            const uint4 xv3 = *reinterpret_cast<const uint4*>(
                x + static_cast<std::int64_t>(3) * kStride + xoff);

            const float2 f00 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv0.x));
            const float2 f01 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv0.y));
            const float2 f02 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv0.z));
            const float2 f03 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv0.w));
            const float2 f10 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv1.x));
            const float2 f11 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv1.y));
            const float2 f12 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv1.z));
            const float2 f13 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv1.w));
            const float2 f20 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv2.x));
            const float2 f21 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv2.y));
            const float2 f22 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv2.z));
            const float2 f23 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv2.w));
            const float2 f30 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv3.x));
            const float2 f31 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv3.y));
            const float2 f32 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv3.z));
            const float2 f33 =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv3.w));

            acc[0] = fmaf(w[0], f00.x, acc[0]);
            acc[0] = fmaf(w[1], f00.y, acc[0]);
            acc[0] = fmaf(w[2], f01.x, acc[0]);
            acc[0] = fmaf(w[3], f01.y, acc[0]);
            acc[0] = fmaf(w[4], f02.x, acc[0]);
            acc[0] = fmaf(w[5], f02.y, acc[0]);
            acc[0] = fmaf(w[6], f03.x, acc[0]);
            acc[0] = fmaf(w[7], f03.y, acc[0]);
            acc[1] = fmaf(w[0], f10.x, acc[1]);
            acc[1] = fmaf(w[1], f10.y, acc[1]);
            acc[1] = fmaf(w[2], f11.x, acc[1]);
            acc[1] = fmaf(w[3], f11.y, acc[1]);
            acc[1] = fmaf(w[4], f12.x, acc[1]);
            acc[1] = fmaf(w[5], f12.y, acc[1]);
            acc[1] = fmaf(w[6], f13.x, acc[1]);
            acc[1] = fmaf(w[7], f13.y, acc[1]);
            acc[2] = fmaf(w[0], f20.x, acc[2]);
            acc[2] = fmaf(w[1], f20.y, acc[2]);
            acc[2] = fmaf(w[2], f21.x, acc[2]);
            acc[2] = fmaf(w[3], f21.y, acc[2]);
            acc[2] = fmaf(w[4], f22.x, acc[2]);
            acc[2] = fmaf(w[5], f22.y, acc[2]);
            acc[2] = fmaf(w[6], f23.x, acc[2]);
            acc[2] = fmaf(w[7], f23.y, acc[2]);
            acc[3] = fmaf(w[0], f30.x, acc[3]);
            acc[3] = fmaf(w[1], f30.y, acc[3]);
            acc[3] = fmaf(w[2], f31.x, acc[3]);
            acc[3] = fmaf(w[3], f31.y, acc[3]);
            acc[3] = fmaf(w[4], f32.x, acc[3]);
            acc[3] = fmaf(w[5], f32.y, acc[3]);
            acc[3] = fmaf(w[6], f33.x, acc[3]);
            acc[3] = fmaf(w[7], f33.y, acc[3]);
        }
    }

#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
        float a = acc[tt];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { a += __shfl_down_sync(0xffffffffu, a, off); }
        if (lane == 0) { s_part[part][tt] = a; }
    }

    __syncthreads();

    if (part == 0 && lane < 4) {
        const float sum = s_part[0][lane] + s_part[1][lane];
        out[static_cast<std::int64_t>(lane) * n + row] = __float2bfloat16(sum);
    }
}

template <class SC, int kFullSlabs, int kStride>
__launch_bounds__(128, 10) __global__ void linear_rowsplit_gemm_smallt_kernel_direct_split4_q5_t4(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t n, std::int32_t k, std::int32_t t,
    std::int32_t padded_k, std::int32_t full_slabs) {
    static_assert(std::is_same_v<SC, Q5Smallt>, "direct split4 small-T kernel is Q5-only");
    static_assert(kFullSlabs > 0 && kStride > 0, "direct split4 requires exact positive shape");
    (void)full_slabs;
    (void)k;
    (void)t;

    __shared__ float s_part[4][4];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int chunk = static_cast<int>(threadIdx.x) >> 5;
    const int row   = static_cast<int>(blockIdx.x);
    if (row >= n) { return; }

    const int           kg_padded = padded_k / Q5Codec::kGroupK;
    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row =
        high + static_cast<std::int64_t>(row) * kg_padded * Q5Smallt::kHighBytesPerGroup;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;

    float acc[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (int s = 0; s < kFullSlabs; ++s) {
        const std::uint8_t* code_phase =
            code_row + static_cast<std::int64_t>(s) * 512 + chunk * 128 + lane * 4;
        const std::uint8_t* high_phase =
            high_row + static_cast<std::int64_t>(s) * 128 + chunk * 32 + lane;
        const int group_in_slab = chunk * 4 + (lane >> 3);
        std::uint32_t scale_bits = 0;
        if ((lane & 7) == 0) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + (static_cast<std::int64_t>(s) * 16 + group_in_slab) * 2);
        }
        scale_bits = __shfl_sync(0xffffffffu, scale_bits, lane & ~7);

        const std::uint32_t word  = *reinterpret_cast<const std::uint32_t*>(code_phase);
        const std::uint32_t hc    = static_cast<std::uint32_t>(*high_phase) ^ 0xffu;
        const float         scale = __half2float(__ushort_as_half(scale_bits));
        const __half2       bias  = __half2half2(__ushort_as_half(0x6410)); // 1040.0
        float               w[8];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            std::uint32_t bits = ((word >> (4 * p)) & 0x000f000fu) | 0x64006400u;
            bits |= (((hc >> p) & 1u) << 4) | (((hc >> (p + 4)) & 1u) << 20);
            const __half2 h = __hsub2(*reinterpret_cast<const __half2*>(&bits), bias);
            const float2  f = __half22float2(h);
            w[p]            = f.x * scale;
            w[p + 4]        = f.y * scale;
        }

        const std::int64_t xoff =
            static_cast<std::int64_t>(s) * 1024 + chunk * 256 + lane * 8;
        const uint4 xv0 = *reinterpret_cast<const uint4*>(x + xoff);
        const uint4 xv1 = *reinterpret_cast<const uint4*>(x + kStride + xoff);
        const uint4 xv2 =
            *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(2) * kStride + xoff);
        const uint4 xv3 =
            *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(3) * kStride + xoff);

        const float2 f00 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv0.x));
        const float2 f01 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv0.y));
        const float2 f02 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv0.z));
        const float2 f03 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv0.w));
        const float2 f10 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv1.x));
        const float2 f11 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv1.y));
        const float2 f12 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv1.z));
        const float2 f13 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv1.w));
        const float2 f20 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv2.x));
        const float2 f21 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv2.y));
        const float2 f22 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv2.z));
        const float2 f23 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv2.w));
        const float2 f30 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv3.x));
        const float2 f31 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv3.y));
        const float2 f32 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv3.z));
        const float2 f33 =
            __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv3.w));

        acc[0] = fmaf(w[0], f00.x, acc[0]);
        acc[0] = fmaf(w[1], f00.y, acc[0]);
        acc[0] = fmaf(w[2], f01.x, acc[0]);
        acc[0] = fmaf(w[3], f01.y, acc[0]);
        acc[0] = fmaf(w[4], f02.x, acc[0]);
        acc[0] = fmaf(w[5], f02.y, acc[0]);
        acc[0] = fmaf(w[6], f03.x, acc[0]);
        acc[0] = fmaf(w[7], f03.y, acc[0]);
        acc[1] = fmaf(w[0], f10.x, acc[1]);
        acc[1] = fmaf(w[1], f10.y, acc[1]);
        acc[1] = fmaf(w[2], f11.x, acc[1]);
        acc[1] = fmaf(w[3], f11.y, acc[1]);
        acc[1] = fmaf(w[4], f12.x, acc[1]);
        acc[1] = fmaf(w[5], f12.y, acc[1]);
        acc[1] = fmaf(w[6], f13.x, acc[1]);
        acc[1] = fmaf(w[7], f13.y, acc[1]);
        acc[2] = fmaf(w[0], f20.x, acc[2]);
        acc[2] = fmaf(w[1], f20.y, acc[2]);
        acc[2] = fmaf(w[2], f21.x, acc[2]);
        acc[2] = fmaf(w[3], f21.y, acc[2]);
        acc[2] = fmaf(w[4], f22.x, acc[2]);
        acc[2] = fmaf(w[5], f22.y, acc[2]);
        acc[2] = fmaf(w[6], f23.x, acc[2]);
        acc[2] = fmaf(w[7], f23.y, acc[2]);
        acc[3] = fmaf(w[0], f30.x, acc[3]);
        acc[3] = fmaf(w[1], f30.y, acc[3]);
        acc[3] = fmaf(w[2], f31.x, acc[3]);
        acc[3] = fmaf(w[3], f31.y, acc[3]);
        acc[3] = fmaf(w[4], f32.x, acc[3]);
        acc[3] = fmaf(w[5], f32.y, acc[3]);
        acc[3] = fmaf(w[6], f33.x, acc[3]);
        acc[3] = fmaf(w[7], f33.y, acc[3]);
    }

#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
        float a = acc[tt];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { a += __shfl_down_sync(0xffffffffu, a, off); }
        if (lane == 0) { s_part[chunk][tt] = a; }
    }

    __syncthreads();

    if (chunk == 0 && lane < 4) {
        float sum = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) { sum += s_part[p][lane]; }
        out[static_cast<std::int64_t>(lane) * n + row] = __float2bfloat16(sum);
    }
}

// full_slabs is computed on the host: k/1024 when k % 8 == 0 and x is 16-byte
// aligned, else 0 (everything runs through the scalar tail).
template <class SC, int kTt, int kRowsPerBlock, int kStages>
__global__ void linear_rowsplit_gemm_smallt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
    __nv_bfloat16* __restrict__ out, std::int32_t n, std::int32_t k, std::int32_t t,
    std::int32_t padded_k, std::int32_t full_slabs) {
    using Codec                = typename SC::Codec;
    constexpr int kPrefetch    = kStages - 1;
    constexpr int kHighU4Alloc = SC::kHighU4 > 0 ? SC::kHighU4 : 1;

    __shared__ __align__(16) uint4         s_nib[kRowsPerBlock][kStages][SC::kNibU4];
    __shared__ __align__(16) uint4         s_hi[kRowsPerBlock][kStages][kHighU4Alloc];
    __shared__ __align__(16) std::uint32_t s_sc[kRowsPerBlock][kStages][SC::kScaleU32];

    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row  = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    if (row >= n) { return; }
    const int col0  = static_cast<int>(blockIdx.y) * kTt;
    const int ncols = min(kTt, t - col0);

    const int           kg_padded = padded_k / Codec::kGroupK;
    const std::uint8_t* code_row  = codes + static_cast<std::int64_t>(row) * kg_padded * 32;
    const std::uint8_t* high_row =
        SC::kHighBytesPerGroup > 0
            ? high + static_cast<std::int64_t>(row) * kg_padded * SC::kHighBytesPerGroup
            : nullptr;
    const std::uint8_t*  scale_row = scales + static_cast<std::int64_t>(row) * kg_padded * 2;
    const __nv_bfloat16* x0        = x + static_cast<std::int64_t>(col0) * k;

    float acc[kTt];
#pragma unroll
    for (int i = 0; i < kTt; ++i) { acc[i] = 0.0f; }

#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < full_slabs) {
            smallt_issue_slab<SC>(s_nib[warp][p], s_hi[warp][p], s_sc[warp][p], code_row,
                                  high_row, scale_row, p, lane);
        } else {
            __pipeline_commit();
        }
    }

#pragma unroll 1
    for (int s = 0; s < full_slabs; ++s) {
        const int fetch = s + kPrefetch;
        if (fetch < full_slabs) {
            const int buf = fetch % kStages;
            smallt_issue_slab<SC>(s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], code_row,
                                  high_row, scale_row, fetch, lane);
        } else {
            __pipeline_commit();
        }
        __pipeline_wait_prior(kPrefetch);
        __syncwarp();

        const int buf = s % kStages;
        smallt_consume_slab<SC, kTt>(x0, static_cast<std::int64_t>(s) * 1024, k, ncols,
                                     s_nib[warp][buf], s_hi[warp][buf], s_sc[warp][buf], lane,
                                     acc);
        __syncwarp();
    }

    // Scalar tail: remaining groups read global memory directly, masked at k.
    const int g0      = (full_slabs * 1024) / Codec::kGroupK;
    const int kg_used = (k + Codec::kGroupK - 1) / Codec::kGroupK;
    for (int g = g0; g < kg_used; ++g) {
        const int kk = g * Codec::kGroupK + lane * 2;
        if (kk >= k) { continue; }
        float w0 = 0.0f;
        float w1 = 0.0f;
        Codec::load_pair(codes, high, scales, static_cast<std::int64_t>(row) * kg_padded + g,
                         lane, w0, w1);
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            if (tt < ncols) {
                const std::int64_t xb = static_cast<std::int64_t>(tt) * k + kk;
                acc[tt] = fmaf(w0, __bfloat162float(x0[xb]), acc[tt]);
                if (kk + 1 < k) { acc[tt] = fmaf(w1, __bfloat162float(x0[xb + 1]), acc[tt]); }
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        if (tt >= ncols) { continue; }
        float a = acc[tt];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) { a += __shfl_down_sync(0xffffffffu, a, off); }
        if (lane == 0) {
            out[static_cast<std::int64_t>(col0 + tt) * n + row] = __float2bfloat16(a);
        }
    }
}

} // namespace qus::kernels::detail
