#pragma once

#include "ops/common/mma.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

// Stable CUDA mechanics shared by semantically separate fused kernels. Format
// backends own their codecs, kernel bodies, schedules, and route selection.
template <int BM_, int BN_, int BK_, int WM_, int WN_, int STAGES_, int MIN_BLOCKS_ = 1,
          bool FRAG_DBUF_ = true, bool CG_LOAD_ = false, bool SCALE_PAIR_LOAD_ = false>
struct GemmCfg {
    static constexpr int BM               = BM_;
    static constexpr int BN               = BN_;
    static constexpr int BK               = BK_;
    static constexpr int WM               = WM_;
    static constexpr int WN               = WN_;
    static constexpr int STAGES           = STAGES_;
    static constexpr int MIN_BLOCKS       = MIN_BLOCKS_;
    static constexpr int WARPS_M          = BM_ / WM_;
    static constexpr int WARPS_N          = BN_ / WN_;
    static constexpr int WARPS            = WARPS_M * WARPS_N;
    static constexpr int THREADS          = WARPS * 32;
    static constexpr int MT               = WM_ / 16;
    static constexpr int NT               = WN_ / 8;
    static constexpr bool FRAG_DBUF       = FRAG_DBUF_;
    static constexpr bool CG_LOAD         = CG_LOAD_;
    static constexpr bool SCALE_PAIR_LOAD = SCALE_PAIR_LOAD_;

    static constexpr int GROUPS_PER_BK = BK_ / 64;
    static constexpr int SCALE_BYTES   = SCALE_PAIR_LOAD_ ? 4 : 2;

    // Upper-bound raw quant staging with a 16-byte high plane. Individual fused
    // kernels define their exact shared arrays.
    static constexpr int smem_est_bytes = (BM_ * BK_) * 2 + STAGES_ * (BN_ * BK_) * 2 +
                                          STAGES_ * (BM_ * GROUPS_PER_BK * (32 + 16 + SCALE_BYTES));

    static_assert(BK_ % 64 == 0, "GemmCfg: BK must be a multiple of 64");
    static_assert(STAGES_ >= 2, "GemmCfg: cp.async pipeline requires at least two stages");
    static_assert(BM_ % WM_ == 0 && BN_ % WN_ == 0,
                  "GemmCfg: block tile must divide into warp tiles");
    static_assert(WM_ % 16 == 0 && WN_ % 8 == 0, "GemmCfg: warp tile must use m16n8 multiples");
    static_assert(WARPS >= 1 && THREADS <= 1024, "GemmCfg: invalid thread count");
    static_assert(smem_est_bytes <= 48 * 1024, "GemmCfg: staged shared exceeds 48 KiB");
};

template <int Bytes, class Cfg>
__device__ __forceinline__ void gemm_cp_async(void* dst, const void* src) {
    if constexpr (Cfg::CG_LOAD && Bytes == 16) {
        cp_async<Bytes, Cache::cg>(dst, src);
    } else {
        cp_async<Bytes>(dst, src);
    }
}

__device__ __forceinline__ int gemm_swz64(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

} // namespace ninfer::ops::detail
