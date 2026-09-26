#pragma once
#include "ops/linear/fp8/fp8_geometry.h"
#include "ops/common/memory.cuh"
#include <cuda_bf16.h>

namespace ninfer::ops::detail {
enum class Fp8CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Fp8SimtActivationAccess : std::uint8_t {
    TokenPacked,
    SharedPhase,
};

enum class Fp8SimtBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Fp8CodeCache CodeCache, int PhaseUnroll, int MinBlocksPerSm>
struct Fp8A16GemvSchedule {
    static constexpr int kStaticK     = 0;
    static constexpr int kBlockTokens = 1;
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kBlockRows         = WarpsPerCta * RowsPerWarp;
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int TokenTile, int AccumulatorChains,
          Fp8SimtActivationAccess ActivationAccess, Fp8CodeCache CodeCache, int PhaseUnroll,
          Fp8SimtBlockOrder BlockOrder, int MinBlocksPerSm>
struct Fp8A16SimtSchedule {
    static constexpr int kStaticK       = 0;
    static constexpr int kTokenCapacity = 0;
    static constexpr bool kExactTokens  = false;
    static constexpr int kBlockTokens   = TokenTile;
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kBlockRows         = WarpsPerCta * RowsPerWarp;
};

enum class Fp8MmaFragmentPipeline : std::uint8_t {
    Serial,
    PingPong,
};

enum class Fp8MmaRaster : std::uint8_t {
    RowFast,
    TokenFast,
    Grouped,
};

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int ActivationStages, int MinBlocksPerSm, Cache WeightCache = Cache::cg,
          Cache ActivationCache                   = Cache::cg,
          Fp8MmaFragmentPipeline FragmentPipeline = Fp8MmaFragmentPipeline::PingPong>
struct Fp8A16MmaSchedule {
    static constexpr int kStaticK           = 0;
    static constexpr auto kFragmentPipeline = FragmentPipeline;
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kBlockK            = BlockK;
    static constexpr int kWarpRows          = WarpRows;
    static constexpr int kWarpTokens        = WarpTokens;
    static constexpr int kActivationStages  = ActivationStages;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;

    static constexpr int kWarpsRows   = kBlockRows / kWarpRows;
    static constexpr int kWarpsTokens = kBlockTokens / kWarpTokens;
    static constexpr int kWarps       = kWarpsRows * kWarpsTokens;
    static constexpr int kThreads     = kWarps * 32;
    static constexpr int kMmaRows     = kWarpRows / 16;
    static constexpr int kMmaTokens   = kWarpTokens / 8;
    static constexpr int kMmaK        = kBlockK / 16;
    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kBlockRows * kBlockK;

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert((kBlockRows % kWarpRows) == 0 && (kBlockTokens % kWarpTokens) == 0);
    static_assert((kWarpRows % 16) == 0 && (kWarpTokens % 8) == 0);
    static_assert(kBlockK == 64 || kBlockK == 128);
    static_assert(kActivationStages == 1 || kActivationStages == 2);
    static_assert(kMinBlocksPerSm > 0);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
};

template <int BlockTokens, int BlockRows, int BlockK, int WarpsTokens, int WarpsRows, int Stages,
          int MinBlocksPerSm, Cache WeightCache, Cache ActivationCache,
          Fp8MmaFragmentPipeline FragmentPipeline, Fp8MmaRaster Raster, int RasterGroupRows = 1>
struct Fp8A8MmaSchedule {
    static constexpr int kStaticK                             = 0;
    static constexpr int kBlockTokens                         = BlockTokens;
    static constexpr int kBlockRows                           = BlockRows;
    static constexpr int kBlockK                              = BlockK;
    static constexpr int kWarpsTokens                         = WarpsTokens;
    static constexpr int kWarpsRows                           = WarpsRows;
    static constexpr int kStages                              = Stages;
    static constexpr int kMinBlocksPerSm                      = MinBlocksPerSm;
    static constexpr Cache kWeightCache                       = WeightCache;
    static constexpr Cache kActivationCache                   = ActivationCache;
    static constexpr Fp8MmaFragmentPipeline kFragmentPipeline = FragmentPipeline;
    static constexpr Fp8MmaRaster kRaster                     = Raster;
    static constexpr int kRasterGroupRows                     = RasterGroupRows;

    static constexpr int kWarps          = kWarpsTokens * kWarpsRows;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kWarpTokens     = kBlockTokens / kWarpsTokens;
    static constexpr int kWarpRows       = kBlockRows / kWarpsRows;
    static constexpr int kMmaTokens      = kWarpTokens / 16;
    static constexpr int kMmaRows        = kWarpRows / 8;
    static constexpr int kMmaK           = kBlockK / 32;
    static constexpr int kSegmentsPerRow = kBlockK / 16;
    static constexpr int kSharedBytes =
        (kStages * (kBlockTokens + kBlockRows) * kBlockK > kBlockTokens * (kBlockRows + 8) * 2
             ? kStages * (kBlockTokens + kBlockRows) * kBlockK
             : kBlockTokens * (kBlockRows + 8) * 2);

    static_assert(kBlockTokens > 0 && kBlockRows > 0 && kBlockK > 0);
    static_assert((kBlockTokens % kWarpsTokens) == 0 && (kBlockRows % kWarpsRows) == 0);
    static_assert((kWarpTokens % 16) == 0 && (kWarpRows % 8) == 0);
    static_assert((kBlockK % 32) == 0);
    static_assert((kSegmentsPerRow & (kSegmentsPerRow - 1)) == 0);
    static_assert(kStages >= 2 && kStages <= 8);
    static_assert(kMinBlocksPerSm >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
    static_assert(kRaster != Fp8MmaRaster::Grouped || kRasterGroupRows > 0);
};


enum class Fp8ActivationStage : std::uint8_t { ActiveOnly, PaddedZero };

template <int KWarps, int TileTokens, int MinBlocksPerSm, Cache ActivationCache = Cache::ca,
          Cache WeightCache                  = Cache::cg,
          Fp8ActivationStage ActivationStage = Fp8ActivationStage::ActiveOnly, int Stages = 1>
struct Fp8A16SlicedKMmaSchedule {
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens > 0 && TileTokens % 8 == 0);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
    static constexpr int kStaticK           = 0;
    static constexpr int kTokenCapacity     = TileTokens;
    static constexpr bool kExactTokens      = false;
    static constexpr int kKWarps            = KWarps;
    static constexpr int kBlockTokens       = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kStages            = Stages;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kBlockK            = KWarps * kTileKPerWarp;
    static constexpr int kBlockRows         = 16;
    static constexpr int kRowsPerLoaderWarp = 16 / KWarps;
    static constexpr int kStagingBytes = Stages * (16 * kBlockK + KWarps * TileTokens * 64 * 2);
    static constexpr int kPartialBytes = KWarps * (TileTokens / 8) * 32 * 4 * 4;
    static constexpr int kSharedBytes =
        kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

// Shape instances retain measured compile-time K and token extents; the templates also
// accept dynamic dimensions when no specialization is selected.
template <class Schedule, int K, int Capacity = 0, bool ExactTokens = false>
struct Fp8ScheduleInstance : Schedule {
    static_assert(K > 0 && K % 32 == 0);
    static constexpr int kStaticK       = K;
    static constexpr int kTokenCapacity = Capacity;
    static constexpr bool kExactTokens  = ExactTokens;
};

struct Fp8IdentityRows {
    static constexpr bool kPaired          = false;
    static constexpr bool kContiguousPairs = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int) const { return begin + row; }
};
} // namespace ninfer::ops::detail
