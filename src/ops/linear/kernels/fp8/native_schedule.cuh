#pragma once

#include "ops/linear/kernels/fp8/fp8_schedule.cuh"

namespace ninfer::ops::detail {

template <int BlockTokens, int Warps = 4, int RowsPerWarp = 1, int ValuesPerLane = 8,
          int Chains = 4, int MinBlocks = 1, bool PostScale = false>
struct NativeFp8A16SimtSchedule {
    static constexpr int kBlockTokens    = BlockTokens;
    static constexpr int kWarps          = Warps;
    static constexpr int kRowsPerWarp    = RowsPerWarp;
    static constexpr int kBlockRows      = Warps * RowsPerWarp;
    static constexpr int kValuesPerLane  = ValuesPerLane;
    static constexpr int kChains         = Chains;
    static constexpr bool kPostScale     = PostScale;
    static constexpr int kThreads        = Warps * 32;
    static constexpr int kMinBlocksPerSm = MinBlocks;
    static constexpr int kStaticK        = 0;
    static_assert(BlockTokens > 0 && BlockTokens <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16);
    static_assert(Chains == 1 || Chains == 2 || Chains == 4 || Chains == 8);
};

template <int BlockRows, int BlockTokens, int BlockK = 128, int WarpRows = 16, int WarpTokens = 8,
          int Stages = 2, int MinBlocks = 1, int RasterGroupRows = 8>
struct NativeFp8MmaSchedule {
    static constexpr int kBlockRows       = BlockRows;
    static constexpr int kBlockTokens     = BlockTokens;
    static constexpr int kBlockK          = BlockK;
    static constexpr int kWarpRows        = WarpRows;
    static constexpr int kWarpTokens      = WarpTokens;
    static constexpr int kWarpsRows       = BlockRows / WarpRows;
    static constexpr int kWarpsTokens     = BlockTokens / WarpTokens;
    static constexpr int kThreads         = kWarpsRows * kWarpsTokens * 32;
    static constexpr int kMmaRows         = WarpRows / 16;
    static constexpr int kMmaTokens       = WarpTokens / 8;
    static constexpr int kStages          = Stages;
    static constexpr int kMinBlocksPerSm  = MinBlocks;
    static constexpr int kStaticK         = 0;
    static constexpr int kRasterGroupRows = RasterGroupRows;
    static constexpr bool kPredicatedRows = true;
    static_assert(BlockRows % WarpRows == 0 && BlockTokens % WarpTokens == 0);
    static_assert(WarpRows % 16 == 0 && WarpTokens % 8 == 0);
    static_assert(BlockK == 64 || BlockK == 128 || BlockK == 256);
    static_assert(Stages > 0 && Stages <= 4 && kThreads <= 1024);
};

template <class Schedule, int K>
struct NativeFp8ScheduleInstance : Schedule {
    static constexpr int kStaticK = K;
};

} // namespace ninfer::ops::detail
