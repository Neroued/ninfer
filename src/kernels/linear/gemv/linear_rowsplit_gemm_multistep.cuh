#pragma once

// Warp-per-row multi-step (T>1) row-split low-bit GEMM: out[N,T] = W[N,K] . x[K,T].
//
// This is the prefill/T>1 low-bit linear path. It reads the same row-split weight
// layout the decode GEMV uses (single VRAM copy; the layout is not changed or
// duplicated for prefill), and adapts on-chip:
//   - Each warp owns one output row; blockIdx.y selects a tile of kTt activation
//     columns. Active lanes own codec-dependent group pairs (2*lane, 2*lane+1).
//   - For every K-group the row's weights are dequantized ONCE (Codec::load_pair)
//     and reused across all kTt columns, so weight dequant is amortized over the
//     column tile instead of re-dequantized per token (the naive reference cost).
//   - fp32 accumulate, warp-shuffle reduction, bf16 output. The dequant math is
//     identical to Codec::load_group / the decode kernels.
//
// This is the correctness-first "usable" prefill kernel: still CUDA-core and it
// re-streams the weights once per kTt-column tile, so it is memory-bound at large
// T. The tensor-core LargeT GEMM (P2) and a bandwidth pushdown are follow-ons.

#include "kernels/linear/codec/linear_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace qus::kernels::detail {

template <class Codec, int kTt>
__global__ void linear_rowsplit_gemm_multistep_kernel(const __nv_bfloat16* __restrict__ x,
                                                      const std::uint8_t* __restrict__ codes,
                                                      const std::uint8_t* __restrict__ high,
                                                      const std::uint8_t* __restrict__ scales,
                                                      __nv_bfloat16* __restrict__ out,
                                                      std::int32_t n, std::int32_t k,
                                                      std::int32_t t, std::int32_t padded_k) {
    constexpr int kGroupK  = Codec::kGroupK;
    constexpr int kPairsPG = kGroupK / 2;
    const int     kg       = padded_k / kGroupK;

    const int lane            = static_cast<int>(threadIdx.x) & 31;
    const int warp            = static_cast<int>(threadIdx.x) >> 5;
    const int warps_per_block = static_cast<int>(blockDim.x) >> 5;
    const int row             = static_cast<int>(blockIdx.x) * warps_per_block + warp;
    if (row >= n) { return; }
    const int col0 = static_cast<int>(blockIdx.y) * kTt;

    const auto*        x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const std::int64_t k2 = static_cast<std::int64_t>(k) >> 1; // bf162 pairs per column
    const bool         even_k = (k & 1) == 0;

    float acc[kTt];
#pragma unroll
    for (int i = 0; i < kTt; ++i) { acc[i] = 0.0f; }

    for (int group = 0; group < kg; ++group) {
        if (lane >= kPairsPG) { continue; }
        const int kk = group * kGroupK + lane * 2;
        if (kk >= k) { continue; }
        const std::int64_t gi = static_cast<std::int64_t>(row) * kg + group;
        float              w0 = 0.0f;
        float              w1 = 0.0f;
        Codec::load_pair(codes, high, scales, gi, lane, w0, w1);
        const std::int64_t xi = static_cast<std::int64_t>(group) * kPairsPG + lane; // = kk>>1
#pragma unroll
        for (int tt = 0; tt < kTt; ++tt) {
            const int col = col0 + tt;
            if (col < t) {
                if (even_k) {
                    const float2 xv =
                        __bfloat1622float2(x2[static_cast<std::int64_t>(col) * k2 + xi]);
                    acc[tt] = fmaf(w0, xv.x, acc[tt]);
                    acc[tt] = fmaf(w1, xv.y, acc[tt]);
                } else {
                    const std::int64_t base = static_cast<std::int64_t>(col) * k + kk;
                    acc[tt] = fmaf(w0, __bfloat162float(x[base]), acc[tt]);
                    if (kk + 1 < k) {
                        acc[tt] = fmaf(w1, __bfloat162float(x[base + 1]), acc[tt]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < kTt; ++tt) {
        float a = acc[tt];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_down_sync(0xffffffffu, a, off);
        }
        const int col = col0 + tt;
        if (lane == 0 && col < t) {
            out[static_cast<std::int64_t>(col) * n + row] = __float2bfloat16(a);
        }
    }
}

} // namespace qus::kernels::detail
