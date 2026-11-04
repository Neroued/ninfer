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

} // namespace ninfer::ops::detail::swiglu_ffn
