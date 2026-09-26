#pragma once
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/common/memory.cuh"
#include <cuda.h>
#include <cuda_bf16.h>

namespace ninfer::ops::detail {
enum class Nvfp4ScaleAccess : std::uint8_t {
    StagedRaw,
    Direct,
};

enum class Nvfp4CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Nvfp4SimtActivationAccess : std::uint8_t {
    PairStream,
    TokenPacked,
    SharedPhase,
};

enum class Nvfp4SimtBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int MinBlocksPerSm>
struct Nvfp4A16GemvSchedule {
    // The unrolled K-phase loops are specialized by Nvfp4ScheduleInstance.
    static constexpr int kStaticK     = 0;
    static constexpr int kBlockTokens = 1;
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kBlockRows         = WarpsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane, int TokenTile,
          int AccumulatorChains, Nvfp4SimtActivationAccess ActivationAccess,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int PhaseUnroll,
          Nvfp4SimtBlockOrder BlockOrder, int MinBlocksPerSm>
struct Nvfp4A16SimtSchedule {
    static constexpr int kStaticK       = 0;
    static constexpr int kTokenCapacity = 0;
    static constexpr bool kExactTokens  = false;
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kBlockTokens       = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kBlockRows         = kRowGroupsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

template <int BlockTokens, int BlockRows, int BlockK, int WarpsTokens, int WarpsRows, int Stages,
          int MinBlocksPerSm, Cache WeightCache = Cache::cg, Cache ActivationCache = Cache::cg>
struct Nvfp4A4MmaSchedule {
    static constexpr int kStaticK           = 0;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;
    static_assert(BlockTokens > 0 && (BlockTokens % 16) == 0);
    static_assert(BlockRows > 0 && (BlockRows % 8) == 0);
    static_assert(BlockK >= 64 && (BlockK % 64) == 0);
    static_assert(WarpsTokens > 0 && WarpsRows > 0);
    static_assert((BlockTokens % WarpsTokens) == 0 && ((BlockTokens / WarpsTokens) % 16) == 0);
    static_assert((BlockRows % WarpsRows) == 0 && ((BlockRows / WarpsRows) % 8) == 0);
    static_assert(Stages >= 1 && Stages <= 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kBlockTokens    = BlockTokens;
    static constexpr int kBlockRows      = BlockRows;
    static constexpr int kBlockK         = BlockK;
    static constexpr int kWarpsTokens    = WarpsTokens;
    static constexpr int kWarpsRows      = WarpsRows;
    static constexpr int kStages         = Stages;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kWarps          = WarpsTokens * WarpsRows;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kWarpTokens     = BlockTokens / WarpsTokens;
    static constexpr int kWarpRows       = BlockRows / WarpsRows;
    static constexpr int kMmaTokens      = kWarpTokens / 16;
    static constexpr int kMmaRows        = kWarpRows / 8;
    static constexpr int kK64PerStage    = BlockK / 64;
    static constexpr int kCodeRowBytes   = BlockK / 2;
    static constexpr int kSegmentsPerRow = kCodeRowBytes / 16;
    static constexpr int kStagingBytes =
        Stages * (BlockTokens + BlockRows) * (BlockK / 2 + BlockK / 16);
    static constexpr int kOutputBytes = BlockTokens * (BlockRows + 8) * 2;
    static constexpr int kSharedBytes = kStagingBytes > kOutputBytes ? kStagingBytes : kOutputBytes;
    static_assert(kThreads <= 1024 && kSharedBytes <= 99 * 1024);
    static_assert((kSegmentsPerRow & (kSegmentsPerRow - 1)) == 0);
};

template <int BlockTokens, int Stages, int MinBlocksPerSm,
          CUtensorMapL2promotion WeightCodePromotion = CU_TENSOR_MAP_L2_PROMOTION_NONE>
struct Nvfp4A4TmaMmaSchedule {
    static constexpr int kStaticK = 0;
    // Each scale tile serves two consecutive K stages. Before a slot is reused,
    // the producer must have waited for the second consumer of the old pair.
    static constexpr int kScaleSlots = (Stages + 2) / 2;
    static constexpr auto kScaleLayout =
        BlockTokens == 128 ? Nvfp4ScaleLayout::Tiled128 : Nvfp4ScaleLayout::Tiled256;
    static_assert(BlockTokens == 128 || BlockTokens == 256);
    static_assert(Stages >= 2 && Stages <= 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr auto kWeightCodePromotion = WeightCodePromotion;

    static constexpr int kBlockTokens      = BlockTokens;
    static constexpr int kBlockRows        = 128;
    static constexpr int kBlockK           = 128;
    static constexpr int kStages           = Stages;
    static constexpr int kWarpsTokens      = 4;
    static constexpr int kWarpsRows        = 2;
    static constexpr int kConsumerWarps    = kWarpsTokens * kWarpsRows;
    static constexpr int kConsumerThreads  = kConsumerWarps * 32;
    static constexpr int kProducerThreads  = BlockTokens == 256 ? 128 : 32;
    static constexpr int kThreads          = kConsumerThreads + kProducerThreads;
    static constexpr int kWarpTokens       = kBlockTokens / kWarpsTokens;
    static constexpr int kWarpRows         = kBlockRows / kWarpsRows;
    static constexpr int kMmaTokens        = kWarpTokens / 16;
    static constexpr int kMmaRows          = kWarpRows / 8;
    static constexpr int kK64PerStage      = 2;
    static constexpr int kScaleWordsPerRow = 4;
    static constexpr int kCodeRowBytes     = 64;
    static constexpr int kMinBlocksPerSm   = MinBlocksPerSm;

    // One scale tile is one shared-memory row per token, and it spans two K tiles - which is why
    // the producer fetches it on even k-tiles only. Both facts are assumptions about
    // kNvfp4ScaleTileGroups held elsewhere, so state them where they would break.
    static_assert(kScaleWordsPerRow * 4 == kNvfp4ScaleTileGroups);
    static_assert((kBlockK / 16) * 2 == kNvfp4ScaleTileGroups);
};


enum class Nvfp4MmaFragmentPipeline : std::uint8_t {
    Serial,
    PingPong,
};

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens,
          int ActivationStages, int MinBlocksPerSm, Cache WeightCache = Cache::cg,
          Cache ActivationCache                     = Cache::cg,
          Nvfp4MmaFragmentPipeline FragmentPipeline = Nvfp4MmaFragmentPipeline::PingPong>
struct Nvfp4A16MmaSchedule {
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
        kBlockRows * (kBlockK / 2 + kBlockK / 16);

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert((kBlockRows % kWarpRows) == 0 && (kBlockTokens % kWarpTokens) == 0);
    static_assert((kWarpRows % 16) == 0 && (kWarpTokens % 8) == 0);
    static_assert(kBlockK == 64 || kBlockK == 128);
    static_assert(kActivationStages == 1 || kActivationStages == 2);
    static_assert(kMinBlocksPerSm > 0);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
};

enum class Nvfp4ActivationStage : std::uint8_t { ActiveOnly, PaddedZero };

template <int KWarps, int TileTokens, int MinBlocksPerSm, Cache ActivationCache = Cache::ca,
          Cache WeightCache                    = Cache::cg,
          Nvfp4ActivationStage ActivationStage = Nvfp4ActivationStage::ActiveOnly, int Stages = 1>
struct Nvfp4A16SlicedKMmaSchedule {
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
    static constexpr int kStagingBytes =
        Stages * (16 * (kBlockK / 2 + kBlockK / 16) + KWarps * TileTokens * 64 * 2);
    static constexpr int kPartialBytes = KWarps * (TileTokens / 8) * 32 * 4 * 4;
    static constexpr int kSharedBytes =
        kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

// Shape instances retain measured compile-time K and token extents; the templates also
// accept dynamic dimensions when no specialization is selected.
template <class Schedule, int K, int Capacity = 0, bool ExactTokens = false>
struct Nvfp4ScheduleInstance : Schedule {
    static_assert(K > 0 && K % 64 == 0);
    static constexpr int kStaticK       = K;
    static constexpr int kTokenCapacity = Capacity;
    static constexpr bool kExactTokens  = ExactTokens;
};

struct Nvfp4IdentityRows {
    static constexpr bool kPaired     = false;
    static constexpr bool kContiguous = true;

    __device__ __forceinline__ int weight_row(int begin, int row, int) const { return begin + row; }
};
} // namespace ninfer::ops::detail
