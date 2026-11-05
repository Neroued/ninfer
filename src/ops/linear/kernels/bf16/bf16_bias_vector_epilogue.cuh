#pragma once

#include "ops/linear/kernels/bf16/bf16_mma_epilogue.cuh"

namespace ninfer::ops::detail {

// Add each row's bias to all FP32 fragments before the vector store rounds once.
// OutputRows binds the complete-row geometry of the consuming shape.
template <int OutputRows>
struct Bf16BiasVectorEpilogue : Bf16TmaVectorEpilogue {
    const __nv_bfloat16* bias;

    template <class S, bool Full, class Output>
    __device__ __forceinline__ void finish_tile(const Output& output, unsigned char* scratch,
                                                float (&accum)[S::kMmaRows][S::kMmaTokens][4],
                                                int row_begin, int token_begin, int rows,
                                                int tokens) const {
        static_assert(OutputRows % S::kBlockRows == 0 && S::kProducerThreads == 0);
        const int warp = threadIdx.x / 32, lane = threadIdx.x & 31;
        const int warp_row = warp / S::kWarpsTokens;
        for (int mi = 0; mi < S::kMmaRows; ++mi) {
            const int row      = row_begin + warp_row * S::kWarpRows + mi * 16 + (lane >> 2);
            const float first  = __bfloat162float(bias[row]);
            const float second = __bfloat162float(bias[row + 8]);
            for (int ni = 0; ni < S::kMmaTokens; ++ni) {
                accum[mi][ni][0] += first;
                accum[mi][ni][1] += first;
                accum[mi][ni][2] += second;
                accum[mi][ni][3] += second;
            }
        }
        const auto destination = linear_output_tile<S::kBlockRows>(output, row_begin);
        if (Full || tokens - token_begin >= S::kBlockTokens)
            Bf16TmaVectorEpilogue::template finish_tile<S, true, false>(
                destination, scratch, accum, row_begin, token_begin, rows, tokens);
        else
            Bf16TmaVectorEpilogue::template finish_tile<S, false, false>(
                destination, scratch, accum, row_begin, token_begin, rows, tokens);
    }
};

} // namespace ninfer::ops::detail
