#pragma once

// Row-blocked Q5 small-T kernel for the fused Q4/Q5 input projections.
//
// Why this exists. The warp-per-row small-T kernel next door streams the weights of one output row
// per warp and reads the activations straight from global (L2) memory. That is the right shape when
// the weights dominate, and at T=1 they do. From about T=4 upward it stops being true: a warp-per-row
// block moves `rows * T * K * 2` bytes of activation for one K pass while the weights are only
// `rows * K * 5/8`, so at T=8 the activation traffic is 25x the weight traffic in bytes. Fitting the
// measured small-T times gives a *logical* effective activation-load bandwidth of about 12.7 TB/s
// for the GDN Q5 parent (12288x5120) -- a derived figure from bytes over time, not a counter
// reading, and L1/L2 hits mean it is not the throughput of one physical interface. Every row
// re-reads the same activation tile, so the fix is not a smaller tile but reuse: stage one
// activation slab in shared memory per block and let all kRowsPerBlock warps read it. That divides
// the repeated activation loads by kRowsPerBlock, and at T=7/8 this shape measured the fastest of
// the candidates tried (62.7 us at T=8 against 93.4 us for the row-split SIMT).
//
// Structure (deliberately the sibling kernel's, plus the staged slab):
//   - one warp owns one output row; blockIdx.y selects a tile of kTt activation columns;
//   - the weights of every row in the block are staged through shared memory with a cp.async
//     pipeline, exactly as the warp-per-row kernel stages them;
//   - one 1024-value slab of the activation tile is staged in shared memory by the whole block and
//     read by every warp for every row, so the global activation reads are amortized over
//     kRowsPerBlock rows;
//   - dequantization is the same fp16-mantissa trick as the sibling kernel, and the group scale is
//     applied per 8-value chunk;
//   - every thread reaches every barrier: a row outside the grid stores nothing but still joins the
//     pipeline, so the block-wide barriers are uniform.
//
// Tails: full 1024-value slabs require k % 8 == 0 (16-byte aligned activation columns) and cover
// [0, k/1024*1024); the remaining groups use the sibling kernel's scalar per-pair path reading global
// memory directly, masked at the k boundary. Weights in [k, padded_k) are never used.

#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

// Store epilogue for the fused input projections: two output tensors with their own row strides and
// a compile-time seam, plus the live column count (a kTt-wide tile may hold fewer valid columns).
// `col0` is the tile's first column in the output; the tile's columns are not necessarily the whole
// token range, so the output column is col0 + token.
struct Q5RowBlockStoreEpilogue {
    template <bool SplitOutput, int SplitRow, int Tokens>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16* out, __nv_bfloat16* out_tail, std::int32_t n, std::int32_t out_ld,
               std::int32_t row, std::int32_t col0, std::int32_t ncols,
               const float (&values)[Tokens]) const {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            if (token >= ncols) { continue; }
            const std::int64_t col = col0 + token;
            if constexpr (SplitOutput) {
                if (row < SplitRow) {
                    out[col * out_ld + row] = __float2bfloat16(values[token]);
                } else {
                    out_tail[col * (n - SplitRow) + row - SplitRow] =
                        __float2bfloat16(values[token]);
                }
            } else {
                out[col * out_ld + row] = __float2bfloat16(values[token]);
            }
        }
    }
};

template <class SC, int kTt, int kRowsPerBlock, int kStages, bool SplitOutput = false,
          int SplitRow = 0, class Epilogue = Q5RowBlockStoreEpilogue>
__launch_bounds__(kRowsPerBlock * 32) __global__
    void q5_rowsplit_rowblock_small_t_kernel(
        const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
        const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales,
        __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ out_tail, std::int32_t n,
        std::int32_t out_ld, std::int32_t k, std::int32_t t, std::int32_t padded_k,
        std::int32_t full_slabs, Epilogue epilogue = {}) {
    using Codec                = typename SC::Codec;
    constexpr int kPrefetch    = kStages - 1;
    constexpr int kHighU4Alloc = SC::kHighU4 > 0 ? SC::kHighU4 : 1;
    constexpr int kSlabK       = 1024;
    constexpr int kVecsPerCol  = kSlabK / 8;
    constexpr int kThreads     = kRowsPerBlock * 32;
    static_assert(kTt >= 1 && kTt <= 32, "row-block small-T requires a bounded column tile");
    static_assert(kRowsPerBlock >= 1 && kRowsPerBlock <= 32, "row-block small-T block size");
    static_assert(kStages >= 2, "row-block small-T requires a cp.async pipeline");
    static_assert(!SplitOutput || SplitRow > 0,
                  "split-output row-block small-T requires a positive compile-time seam");

    __shared__ __align__(16) __nv_bfloat16 x_sh[kTt * kSlabK];
    __shared__ __align__(16) uint4 s_nib[kRowsPerBlock][kStages][SC::kNibU4];
    __shared__ __align__(16) uint4 s_hi[kRowsPerBlock][kStages][kHighU4Alloc];
    __shared__ __align__(16) std::uint32_t s_sc[kRowsPerBlock][kStages][SC::kScaleU32];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int row   = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    const int col0  = static_cast<int>(blockIdx.y) * kTt;
    const int ncols = t - col0 < kTt ? t - col0 : kTt;
    // A row outside the grid still joins every barrier; only its loads and stores are dropped.
    const bool live = row < n && ncols > 0;
    const int safe_row = live ? row : 0;

    const int kg_padded = padded_k / Codec::kGroupK;
    const std::uint8_t* code_row = codes + static_cast<std::int64_t>(safe_row) * kg_padded * 32;
    const std::uint8_t* high_row =
        SC::kHighBytesPerGroup > 0
            ? high + static_cast<std::int64_t>(safe_row) * kg_padded * SC::kHighBytesPerGroup
            : nullptr;
    const std::uint8_t* scale_row = scales + static_cast<std::int64_t>(safe_row) * kg_padded * 2;
    const __nv_bfloat16* x0       = x + static_cast<std::int64_t>(col0) * k;

    float acc[kTt];
#pragma unroll
    for (int i = 0; i < kTt; ++i) { acc[i] = 0.0f; }

    const auto issue_weights = [&](int slab) {
        if (live) {
            q5_simt_issue_slab<SC>(s_nib[warp][slab % kStages], s_hi[warp][slab % kStages],
                                   s_sc[warp][slab % kStages], code_row, high_row, scale_row, slab,
                                   lane);
        } else {
            pipe_commit();
        }
    };

#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
        if (p < full_slabs) {
            issue_weights(p);
        } else {
            pipe_commit();
        }
    }

#pragma unroll 1
    for (int s = 0; s < full_slabs; ++s) {
        const int fetch = s + kPrefetch;
        if (fetch < full_slabs) {
            issue_weights(fetch);
        } else {
            pipe_commit();
        }

        // Stage this slab's activation tile once for the whole block. The barrier before it guarantees
        // no warp is still reading the previous slab; the barrier after it publishes the stores.
        __syncthreads();
        {
            const auto* src = reinterpret_cast<const uint4*>(x0);
            auto* dst       = reinterpret_cast<uint4*>(x_sh);
            const int limit = ncols * kVecsPerCol;
            for (int i = static_cast<int>(threadIdx.x); i < limit; i += kThreads) {
                const int col = i / kVecsPerCol;
                const int vec = i - col * kVecsPerCol;
                dst[i] = src[static_cast<std::int64_t>(col) * (k / 8) + s * kVecsPerCol + vec];
            }
        }
        pipe_wait<kPrefetch>();
        __syncthreads();

#pragma unroll
        for (int c = 0; c < 4; ++c) {
            float w[8];
            SC::dequant_chunk(s_nib[warp][s % kStages], s_hi[warp][s % kStages],
                              s_sc[warp][s % kStages], c, lane, w);
            const std::int64_t xoff = static_cast<std::int64_t>(c) * 256 + lane * 8;
#pragma unroll
            for (int tt = 0; tt < kTt; ++tt) {
                if (tt < ncols) {
                    const uint4 xv = *reinterpret_cast<const uint4*>(&x_sh[tt * kSlabK + xoff]);
                    const float2 f0 = bf16x2_bits_to_float2(xv.x);
                    const float2 f1 = bf16x2_bits_to_float2(xv.y);
                    const float2 f2 = bf16x2_bits_to_float2(xv.z);
                    const float2 f3 = bf16x2_bits_to_float2(xv.w);
                    acc[tt]         = fmaf(w[0], f0.x, acc[tt]);
                    acc[tt]         = fmaf(w[1], f0.y, acc[tt]);
                    acc[tt]         = fmaf(w[2], f1.x, acc[tt]);
                    acc[tt]         = fmaf(w[3], f1.y, acc[tt]);
                    acc[tt]         = fmaf(w[4], f2.x, acc[tt]);
                    acc[tt]         = fmaf(w[5], f2.y, acc[tt]);
                    acc[tt]         = fmaf(w[6], f3.x, acc[tt]);
                    acc[tt]         = fmaf(w[7], f3.y, acc[tt]);
                }
            }
        }
        // The next iteration's issue rewrites the slot this slab was read from, and a 16-byte
        // cp.async copy issued by one lane fills words another lane reads, so the warp has to be
        // converged before that copy is issued. The warp-per-row kernel carries the same fence.
        __syncwarp();
    }

    // Scalar tail: remaining groups read global memory directly, masked at k. Identical to the
    // warp-per-row kernel's tail, so a shape with k % 8 != 0 takes the same path in both.
    const int g0      = (full_slabs * kSlabK) / Codec::kGroupK;
    const int kg_used = div_up(k, Codec::kGroupK);
    if (live) {
        for (int g = g0; g < kg_used; ++g) {
            const int kk = g * Codec::kGroupK + lane * 2;
            if (kk >= k) { continue; }
            float w0 = 0.0f;
            float w1 = 0.0f;
            Codec::load_pair(codes, high, scales,
                             static_cast<std::int64_t>(row) * kg_padded + g, lane, w0, w1);
#pragma unroll
            for (int tt = 0; tt < kTt; ++tt) {
                if (tt < ncols) {
                    const std::int64_t xb = static_cast<std::int64_t>(tt) * k + kk;
                    acc[tt]               = fmaf(w0, __bfloat162float(x0[xb]), acc[tt]);
                    if (kk + 1 < k) { acc[tt] = fmaf(w1, __bfloat162float(x0[xb + 1]), acc[tt]); }
                }
            }
        }
    }

    if (!live) { return; }
    float values[kTt];
#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        float a     = acc[tt];
        a           = warp_reduce_sum(a);
        values[tt]  = a;
    }
    if (lane == 0) {
        epilogue.template operator()<SplitOutput, SplitRow, kTt>(out, out_tail, n, out_ld, row, col0,
                                                                 ncols, values);
    }
}

} // namespace ninfer::ops::detail
