#pragma once

#include "ops/common/quant/fp8.cuh"
#include "ops/linear/kernels/mxfp8/a8_plan.h"
#include "ops/linear/swiglu/product_epilogue.cuh"

namespace ninfer::ops::detail::swiglu_ffn {

struct Mxfp8ProductOutput {
    Mxfp8A8Workspace values;
    int rows;
};

// A CTA owns complete G32 product groups. Its FP32 registers pass through shared
// memory only; the global intermediate contains the final MXFP8 codes and scales.
struct Mxfp8ProductEpilogue {
    ClampedSwiGluProduct product;

    template <class S>
    static constexpr int kSharedBytes = S::kBlockRows / 2 * S::kBlockTokens * sizeof(float);

    template <class S, bool Full>
    __device__ void finish_tile(Mxfp8ProductOutput out, unsigned char* scratch,
                                float (&acc)[S::kMmaRows][S::kMmaTokens][4], int row_begin,
                                int token_begin, int rows, int tokens) const {
        constexpr int products = S::kBlockRows / 2;
        static_assert(products % 32 == 0);
        auto* values   = reinterpret_cast<float*>(scratch);
        const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
        const int wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
#pragma unroll
        for (int mi = 0; mi < S::kMmaRows; ++mi) {
            const int local_row = wr * (S::kWarpRows / 2) + mi * 8 + (lane >> 2);
#pragma unroll
            for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                const int local_token = wt * S::kWarpTokens + ni * 8 + (lane & 3) * 2;
                values[local_token * products + local_row] =
                    product(acc[mi][ni][0], acc[mi][ni][2]);
                values[(local_token + 1) * products + local_row] =
                    product(acc[mi][ni][1], acc[mi][ni][3]);
            }
        }
        __syncthreads();
        for (int item = threadIdx.x; item < S::kBlockTokens * products; item += S::kThreads) {
            const int local_token = item / products, local_row = item % products;
            const float value = values[item];
            float maximum     = fabsf(value);
#pragma unroll
            for (int delta = 16; delta; delta >>= 1)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, delta));
            const auto scale = common::fp8_ue8m0_scale(maximum);
            const int row = row_begin + local_row, token = token_begin + local_token;
            if (row < rows / 2 && (Full || token < tokens)) {
                out.values.codes[std::int64_t(token) * out.rows + row] =
                    common::fp8_quant_code(value, scale.inverse_scale);
                if (!lane)
                    out.values.scales[std::int64_t(token) * (out.rows / 32) + row / 32] =
                        scale.scale;
            }
        }
    }
};

// Quantize four products per thread. Eight lanes cooperate on one G32 scale.
// This path requires A8 permission for the down input.
struct Mxfp8PackedProductEpilogue {
    ClampedSwiGluProduct product;
    int logical_tokens;
    template <class C>
    static constexpr int kSharedBytes = C::kBlockRows / 2 * C::kBlockTokens * 4;

    template <class C, bool Full>
    __device__ __forceinline__ void finish_tile(Mxfp8ProductOutput out, unsigned char* raw,
                                                float (&acc)[C::kMmaRows][C::kMmaTokens][4], int rb,
                                                int tb, int rows, int tokens) const {
        constexpr int P = C::kBlockRows / 2;
        static_assert(P % 32 == 0);
        auto* tile    = reinterpret_cast<float*>(raw);
        const int tid = threadIdx.x, warp = tid / 32, lane = tid & 31;
        const int wr = warp / C::kWarpsTokens, wt = warp % C::kWarpsTokens;
        // XOR rows within each G32 group so adjacent token stores use distinct banks.
        auto at   = [&](int r, int t) -> float* { return tile + t * P + (r ^ ((t & 6) << 2)); };
        auto calc = [&](float g, float u) {
            g = fminf(g, product.limit);
            u = fmaxf(-product.limit, fminf(u, product.limit));
            return __fdividef(g, 1.F + __expf(-g)) * u;
        };
        for (int mi = 0; mi < C::kMmaRows; ++mi) {
            int r = wr * (C::kWarpRows / 2) + mi * 8 + (lane >> 2);
            for (int ni = 0; ni < C::kMmaTokens; ++ni) {
                int t   = wt * C::kWarpTokens + ni * 8 + (lane & 3) * 2;
                auto& a = acc[mi][ni];
                // Discarded padding stays finite through scale selection and down MMA.
                *at(r, t)     = tb + t < logical_tokens ? calc(a[0], a[2]) : 1.F;
                *at(r, t + 1) = tb + t + 1 < logical_tokens ? calc(a[1], a[3]) : 1.F;
            }
        }
        __syncthreads();
        for (int i = tid; i < C::kBlockTokens * (P / 4); i += C::kThreads) {
            int t = i / (P / 4), r = i % (P / 4) * 4;
            float4 v      = common::load_vec<float4>(at(r, t));
            float maximum = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
            for (int d = 4; d; d >>= 1)
                maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d, 8));
            auto sf = common::fp8_ue8m0_scale(maximum);
            if (rb + r + 3 < rows / 2 && (Full || tb + t < tokens)) {
                unsigned word =
                    common::fp8_quant_code2(v.x, v.y, sf.inverse_scale) |
                    (static_cast<unsigned>(common::fp8_quant_code2(v.z, v.w, sf.inverse_scale))
                     << 16);
                common::store_vec(out.values.codes + std::int64_t(tb + t) * out.rows + rb + r,
                                  word);
                if (!(lane & 7))
                    out.values.scales[std::int64_t(tb + t) * (out.rows / 32) + (rb + r) / 32] =
                        sf.scale;
            }
        }
    }
};

} // namespace ninfer::ops::detail::swiglu_ffn
