#pragma once

#include "ops/common/cuda/mma.cuh"

namespace ninfer::ops::detail::mhc {
__device__ __forceinline__ int projection_shared_column(int row, int column) {
    return column ^ ((row & 7) * 8);
}

template <int Tokens>
struct ProjectionSchedule {
    static_assert(Tokens == 8 || Tokens == 16 || Tokens == 32);
    static constexpr int kThreads = Tokens * 8, kWarpsTokens = Tokens / 8;
    static constexpr int kMmaRows = 1, kMmaTokens = 1;
};

// Native F has 24 rows. The projection tile uses two 16-row MMA warps per
// group of eight tokens; padding and the reduction epilogue belong to mHC.
template <int Tokens>
__device__ __forceinline__ void projection_mma_stage(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                                     float (&accum)[4], int warp, int lane) {
    constexpr int BK = 64, TokenWarps = Tokens / 8;
    const int row_group = warp / TokenWarps, token_group = warp % TokenWarps;
    unsigned af[2][4], bf[2][2];
    const auto load = [&](int k, int slot) {
        const int ar = row_group * 16 + (lane & 7) + (((lane >> 3) & 1) * 8);
        const int ac = k * 16 + (lane >> 4) * 8;
        const int br = token_group * 8 + (lane & 7);
        const int bc = k * 16 + (((lane >> 3) & 1) * 8);
        common::ldmatrix_x4(af[slot][0], af[slot][1], af[slot][2], af[slot][3],
                            common::smem_addr(a + ar * BK + projection_shared_column(ar, ac)));
        common::ldmatrix_x2(bf[slot][0], bf[slot][1],
                            common::smem_addr(b + br * BK + projection_shared_column(br, bc)));
    };
    load(0, 0);
#pragma unroll
    for (int k = 0; k < BK / 16; ++k) {
        const int slot = k & 1;
        if (k + 1 < BK / 16) load(k + 1, slot ^ 1);
        common::mma_bf16(accum[0], accum[1], accum[2], accum[3], af[slot][0], af[slot][1],
                         af[slot][2], af[slot][3], bf[slot][0], bf[slot][1]);
    }
}
} // namespace ninfer::ops::detail::mhc
