#pragma once

#include "ops/common/host/launch.h"
#include "ops/common/numeric/math.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/kernels/fp8/native_schedule.cuh"

namespace ninfer::ops::detail {

template <class S, bool Full, class Output, class Epilogue>
__device__ __forceinline__ void
native_fp8_finish(Output output, Epilogue epilogue, unsigned char* scratch,
                  float (&acc)[S::kMmaRows][S::kMmaTokens][4], int row_begin, int token_begin,
                  int rows, int tokens) {
    if constexpr (requires {
                      epilogue.template finish_tile<S, Full>(output, scratch, acc, row_begin,
                                                             token_begin, rows, tokens);
                  }) {
        __syncthreads();
        epilogue.template finish_tile<S, Full>(output, scratch, acc, row_begin, token_begin, rows,
                                               tokens);
    } else {
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        const int wr = warp / S::kWarpsTokens, wt = warp % S::kWarpsTokens;
#pragma unroll
        for (int r = 0; r < S::kMmaRows; ++r) {
            const int row = row_begin + wr * S::kWarpRows + r * 16 + (lane >> 2);
#pragma unroll
            for (int t = 0; t < S::kMmaTokens; ++t) {
                const int token  = token_begin + wt * S::kWarpTokens + t * 8 + 2 * (lane & 3);
                const auto& a    = acc[r][t];
                const auto value = make_float4(a[0], a[1], a[2], a[3]);
                if constexpr (requires {
                                  epilogue.store_fragment(output, row, token, value, rows, tokens);
                              }) {
                    if (row - (lane >> 2) + 16 <= rows)
                        linear_finish_mma_fragment<Full, Output, Epilogue, true>(
                            output, epilogue, row, token, value, rows, tokens);
                } else {
                    linear_finish_mma_fragment<Full, Output, Epilogue, false>(
                        output, epilogue, row, token, value, rows, tokens);
                }
            }
        }
    }
}

template <class S>
__device__ __forceinline__ void native_fp8_tile(int linear, int rows, int tokens, int& row_tile,
                                                int& token_tile) {
    const int row_tiles   = (rows + S::kBlockRows - 1) / S::kBlockRows;
    const int token_tiles = (tokens + S::kBlockTokens - 1) / S::kBlockTokens;
    if constexpr (requires { S::kRasterGroupTokens; }) {
        if constexpr (S::kRasterGroupTokens > 0) {
            constexpr int width = S::kRasterGroupTokens;
            int token_first, active_tokens, local;
            // Avoid a final panel with only one column of CTAs.
            if (token_tiles > width && token_tiles % width == 1) {
                const int panels = (token_tiles + width - 1) / width;
                const int base = token_tiles / panels, extra = token_tiles % panels;
                const int big_span = row_tiles * (base + 1);
                if (linear < extra * big_span) {
                    const int panel = linear / big_span;
                    token_first = panel * (base + 1);
                    active_tokens = base + 1;
                    local = linear - panel * big_span;
                } else {
                    const int tail = linear - extra * big_span;
                    const int panel_span = row_tiles * base;
                    const int panel = tail / panel_span;
                    token_first = extra * (base + 1) + panel * base;
                    active_tokens = base;
                    local = tail - panel * panel_span;
                }
            } else {
                const int panel_span = row_tiles * width;
                const int panel = linear / panel_span;
                token_first = panel * width;
                active_tokens = min(width, token_tiles - token_first);
                local = linear - panel * panel_span;
            }
            const int span = S::kRasterGroupRows * active_tokens;
            const int group = local / span, first = group * S::kRasterGroupRows;
            const int active = min(S::kRasterGroupRows, row_tiles - first);
            const int within = local - group * span;
            row_tile = first + within % active;
            token_tile = token_first + within / active;
            return;
        }
    }
    const int span        = S::kRasterGroupRows * token_tiles;
    const int group       = linear / span;
    const int first       = group * S::kRasterGroupRows;
    const int active      = min(S::kRasterGroupRows, row_tiles - first);
    const int within      = linear - group * span;
    row_tile              = first + within % active;
    token_tile            = within / active;
}

template <class S, class Operands>
void validate_native_fp8(const Operands& p) {
    if (!p.x || !p.codes || !p.scales || p.rows <= 0 || p.k <= 0 || p.k % 128 || p.tokens <= 0)
        throw std::invalid_argument("native FP8 contraction requires valid operands and K128");
    if constexpr (requires { p.x_low; })
        if (!p.x_low)
            throw std::invalid_argument("native FP8 compensated input requires its low plane");
    if constexpr (S::kStaticK)
        if (p.k != S::kStaticK)
            throw std::invalid_argument("native FP8 contraction static K mismatch");
}


} // namespace ninfer::ops::detail
