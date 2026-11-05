#pragma once

#include "ops/linear/swiglu/product_epilogue.cuh"

namespace ninfer::ops::detail {

// Compute products unconditionally; tail predicates apply only to the memory writes.
struct SwiGluPredicatedBf16Output {
    LinearBf16Output output;
    int tokens;

    __device__ __forceinline__ void store(int row, int token, float value) const {
        const bool valid          = token < tokens;
        auto* pointer             = output.at(row, valid ? token : 0);
        const unsigned short bits = __bfloat16_as_ushort(__float2bfloat16_rn(value));
        asm volatile(
            "{ .reg .pred p; setp.ne.u32 p, %2, 0; @p st.global.b16 [%0], %1; }" ::"l"(pointer),
            "h"(bits), "r"(int(valid))
            : "memory");
    }

    __device__ __forceinline__ void store_vector(int row, int token, uint4 value) const {
        const bool valid = token < tokens;
        auto* pointer    = output.at(row, valid ? token : 0);
        asm volatile(
            "{ .reg .pred p; setp.ne.u32 p, %5, 0; @p st.global.v4.u32 [%0], {%1,%2,%3,%4}; }" ::
                "l"(pointer),
            "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w), "r"(int(valid))
            : "memory");
    }
};

// Two row warps own the gate and up halves. Preserve FP32 up values across the
// handoff, then round the complete product once into coalesced BF16 output.
template <class Product>
struct BasicSwiGluRowMajorBf16VectorEpilogue {
    Product product;

    template <class S>
    static constexpr int kSharedBytes = S::kBlockRows * S::kBlockTokens * 2 +
                                        ((S::kBlockRows / 2 + 63) / 64 * 64) * S::kBlockTokens * 2;

    template <class S, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(Output output, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row_begin, int token_begin, int rows,
                                                int tokens) const {
        constexpr int Half = S::kBlockRows / 2, BT = S::kBlockTokens;
        constexpr int PaddedRows = (Half + 63) / 64 * 64;
        static_assert(S::kBlockRows == 2 * S::kWarpRows && Half % 32 == 0);
        auto* up       = reinterpret_cast<float*>(scratch);
        auto* result   = reinterpret_cast<__nv_bfloat16*>(scratch + Half * BT * sizeof(float));
        const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
        const int row_warp = warp / S::kWarpsTokens, token_warp = warp % S::kWarpsTokens;
        // Token-major swizzling makes each MMA fragment's 32 lanes use distinct banks.
        const auto up_at = [&](int row, int token) -> float& {
            return up[token * Half + (row ^ ((token & 6) << 2))];
        };
        const auto result_at = [&](int row, int token) {
            return result + token * PaddedRows + (row ^ ((token & 7) << 3));
        };
        if (row_warp == 1) {
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int row = mi * 16 + (lane >> 2);
#pragma unroll
                for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                    const int token       = token_warp * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    const auto& a         = accum[mi][ni];
                    up_at(row, token)     = a[0];
                    up_at(row, token + 1) = a[1];
                    up_at(row + 8, token) = a[2];
                    up_at(row + 8, token + 1) = a[3];
                }
            }
        }
        __syncthreads();
        if (row_warp == 0) {
#pragma unroll
            for (int mi = 0; mi < S::kMmaRows; ++mi) {
                const int row = mi * 16 + (lane >> 2);
#pragma unroll
                for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                    const int token        = token_warp * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                    const auto& a          = accum[mi][ni];
                    *result_at(row, token) = __float2bfloat16_rn(
                        swiglu_product(product, row_begin + row, a[0], up_at(row, token)));
                    *result_at(row, token + 1) = __float2bfloat16_rn(
                        swiglu_product(product, row_begin + row, a[1], up_at(row, token + 1)));
                    *result_at(row + 8, token) = __float2bfloat16_rn(
                        swiglu_product(product, row_begin + row + 8, a[2], up_at(row + 8, token)));
                    *result_at(row + 8, token + 1) = __float2bfloat16_rn(swiglu_product(
                        product, row_begin + row + 8, a[3], up_at(row + 8, token + 1)));
                }
            }
        }
        __syncthreads();
        for (int vector = threadIdx.x; vector < BT * (Half / 8); vector += S::kThreads) {
            const int token = vector / (Half / 8), row = vector % (Half / 8) * 8;
            if (row_begin + row + 7 < rows / 2 && (Full || token_begin + token < tokens))
                output.store_vector(row_begin + row, token_begin + token,
                                    common::load_vec<uint4>(result_at(row, token)));
        }
    }
};

template <class Product, int OutputRows, template <class> class MathEpilogue>
struct BasicSwiGluPaddedBf16Epilogue {
    Product product;
    int logical_tokens;

    template <class S>
    static constexpr int kSharedBytes = MathEpilogue<Product>::template kSharedBytes<S>;

    template <class S, bool Full>
    __device__ __forceinline__ void finish_tile(LinearBf16Output output, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row, int token, int rows, int) const {
        static_assert(2 * OutputRows % S::kBlockRows == 0);
        // Discarded lanes need no meaningful nonlinear result. Keep their values
        // in an ordinary finite range while computing the masked products.
        if (logical_tokens - token < S::kBlockTokens) {
            const int lane       = threadIdx.x & 31;
            const int token_warp = (threadIdx.x / 32) % S::kWarpsTokens;
#pragma unroll
            for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                const int column = token + token_warp * S::kWarpTokens + ni * 8 + 2 * (lane & 3);
                const bool first = column < logical_tokens, second = column + 1 < logical_tokens;
#pragma unroll
                for (int mi = 0; mi < S::kMmaRows; ++mi) {
                    auto& a = accum[mi][ni];
                    a[0]    = first ? a[0] : 1.0f;
                    a[2]    = first ? a[2] : 1.0f;
                    a[1]    = second ? a[1] : 1.0f;
                    a[3]    = second ? a[3] : 1.0f;
                }
            }
        }
        const auto epilogue = MathEpilogue<Product>{product};
        epilogue.template finish_tile<S, true>(SwiGluPredicatedBf16Output{output, logical_tokens},
                                               scratch, accum, row, token, rows, logical_tokens);
    }
};

} // namespace ninfer::ops::detail
