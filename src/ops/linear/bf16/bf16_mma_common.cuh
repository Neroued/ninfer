#pragma once
#include "core/device.h"
#include "ops/common/mma.cuh"
#include "ops/linear/bf16/bf16_schedule.cuh"
#include "ops/linear/common/epilogue.cuh"

namespace ninfer::ops::detail {

template <int Bytes, auto Kernel>
int bf16_prepare_shared() {
    static_assert(Bytes <= 99 * 1024);
    if constexpr (Bytes > 48 * 1024) {
        static const auto status =
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Bytes);
        CUDA_CHECK(status);
    }
    return Bytes;
}

template <class Schedule, class Epilogue>
inline constexpr int bf16_epilogue_bytes = [] {
    if constexpr (requires { Epilogue::template kSharedBytes<Schedule>; })
        return Epilogue::template kSharedBytes<Schedule>;
    else
        return 0;
}();

template <class Schedule, class Epilogue>
inline constexpr int bf16_mma_shared_bytes =
    Schedule::kSharedBytes > bf16_epilogue_bytes<Schedule, Epilogue>
        ? Schedule::kSharedBytes
        : bf16_epilogue_bytes<Schedule, Epilogue>;

template <class Schedule>
__device__ __forceinline__ int bf16_mma_shared_col(int row, int col) {
    if constexpr (Schedule::kSwizzle == Bf16MmaSwizzle::Plain) {
        return col;
    } else if constexpr (Schedule::kSwizzle == Bf16MmaSwizzle::Tma128) {
        const int chunk = (row * Schedule::kBlockK + col) / 64;
        return (col & ~63) + ((((col & 63) >> 3) ^ (chunk & 7)) << 3) + (col & 7);
    } else {
        return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
    }
}

template <class Schedule>
__device__ __forceinline__ void
bf16_mma_tile_coordinates(std::int32_t linear, std::int32_t tiles_m, std::int32_t tiles_n,
                          std::int32_t& tile_m, std::int32_t& tile_n) {
    if constexpr (Schedule::kRaster == Bf16MmaRaster::TokenFast) {
        tile_m = linear / tiles_n;
        tile_n = linear - tile_m * tiles_n;
    } else if constexpr (Schedule::kRaster == Bf16MmaRaster::RowFast) {
        tile_n = linear / tiles_m;
        tile_m = linear - tile_n * tiles_m;
    } else {
        constexpr int group_rows      = Schedule::kRasterGroupRows;
        const std::int32_t group_span = group_rows * tiles_n;
        const std::int32_t group      = linear / group_span;
        const std::int32_t first_m    = group * group_rows;
        const std::int32_t active_m   = min(group_rows, tiles_m - first_m);
        const std::int32_t within     = linear - group * group_span;
        tile_m                        = first_m + within % active_m;
        tile_n                        = within / active_m;
    }
}

template <class Schedule>
__device__ __forceinline__ void
bf16_mma_compute_stage(const __nv_bfloat16* As, const __nv_bfloat16* Bs,
                       float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int warp,
                       int lane) {
    constexpr int BK = Schedule::kBlockK, WM = Schedule::kWarpRows, WN = Schedule::kWarpTokens;
    constexpr int MT = Schedule::kMmaRows, NT = Schedule::kMmaTokens, KSUB = BK / 16;
    const int wm = warp / Schedule::kWarpsTokens, wn = warp % Schedule::kWarpsTokens;
    const int a_matrix = lane >> 3, a_inner_row = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row = lane & 7, b_k_offset = ((lane >> 3) & 1) << 3;
    auto load_fragments = [&](int k_step, unsigned(&a_frag)[MT][4], unsigned(&b_frag)[NT][2]) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int row = wm * WM + mi * 16 + a_row_offset;
            const int col = k_step * 16 + a_col_offset;
            ldmatrix_x4(a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        smem_addr(&As[row * BK + bf16_mma_shared_col<Schedule>(row, col)]));
        }
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int row = wn * WN + ni * 8 + b_inner_row;
            const int col = k_step * 16 + b_k_offset;
            ldmatrix_x2(b_frag[ni][0], b_frag[ni][1],
                        smem_addr(&Bs[row * BK + bf16_mma_shared_col<Schedule>(row, col)]));
        }
    };

    if constexpr (Schedule::kFragmentPipeline == Bf16MmaFragmentPipeline::PingPong) {
        unsigned a_frag[2][MT][4];
        unsigned b_frag[2][NT][2];
        load_fragments(0, a_frag[0], b_frag[0]);
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < KSUB) {
                load_fragments(k_step + 1, a_frag[slot ^ 1], b_frag[slot ^ 1]);
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2], accum[mi][ni][3],
                             a_frag[slot][mi][0], a_frag[slot][mi][1], a_frag[slot][mi][2],
                             a_frag[slot][mi][3], b_frag[slot][ni][0], b_frag[slot][ni][1]);
                }
            }
        }
    } else {
        unsigned a_frag[MT][4];
        unsigned b_frag[NT][2];
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            load_fragments(k_step, a_frag, b_frag);
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    mma_bf16(accum[mi][ni][0], accum[mi][ni][1], accum[mi][ni][2], accum[mi][ni][3],
                             a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                             b_frag[ni][0], b_frag[ni][1]);
                }
            }
        }
    }
}

template <bool FullTokens, class Output, class Epilogue>
__device__ __forceinline__ void bf16_finish_fragment(const Output& output, const Epilogue& epilogue,
                                                     int row, int token, float4 value, int rows,
                                                     int tokens) {
    if constexpr (requires { epilogue.store_fragment(output, row, token, value, rows, tokens); }) {
        // A fragment consumer receives the atom origin and owns its token-tail predicates.
        epilogue.store_fragment(output, row, token, value, rows, tokens);
    } else {
        if (FullTokens || token < tokens) {
            output.store(row, token, epilogue.apply(row, token, value.x));
            output.store(row + 8, token, epilogue.apply(row + 8, token, value.z));
        }
        if (FullTokens || token + 1 < tokens) {
            output.store(row, token + 1, epilogue.apply(row, token + 1, value.y));
            output.store(row + 8, token + 1, epilogue.apply(row + 8, token + 1, value.w));
        }
    }
}

template <class Schedule, bool FullTokens, class Output, class Epilogue>
__device__ __forceinline__ void
bf16_finish_mma_tile(const Output& output, const Epilogue& epilogue, unsigned char* scratch,
                     float (&accum)[Schedule::kMmaRows][Schedule::kMmaTokens][4], int row_begin,
                     int token_begin, int rows, int tokens, int warp, int lane) {
    const auto destination = linear_output_tile<Schedule::kBlockRows>(output, row_begin);
    if constexpr (requires {
                      epilogue.template finish_tile<Schedule, FullTokens>(
                          destination, scratch, accum, row_begin, token_begin, rows, tokens);
                  }) {
        // All staging reads must have finished before a collective consumer reuses scratch.
        __syncthreads();
        epilogue.template finish_tile<Schedule, FullTokens>(destination, scratch, accum, row_begin,
                                                            token_begin, rows, tokens);
    } else {
        const int wm = warp / Schedule::kWarpsTokens, wn = warp % Schedule::kWarpsTokens;
#pragma unroll
        for (int mi = 0; mi < Schedule::kMmaRows; ++mi) {
            const int row = row_begin + wm * Schedule::kWarpRows + mi * 16 + (lane >> 2);
#pragma unroll
            for (int ni = 0; ni < Schedule::kMmaTokens; ++ni) {
                const int token =
                    token_begin + wn * Schedule::kWarpTokens + ni * 8 + 2 * (lane & 3);
                const auto& v = accum[mi][ni];
                bf16_finish_fragment<FullTokens>(destination, epilogue, row, token,
                                                 make_float4(v[0], v[1], v[2], v[3]), rows, tokens);
            }
        }
    }
}
} // namespace ninfer::ops::detail
