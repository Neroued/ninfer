#pragma once

#include "ops/common/memory.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Bf16ActivationAccess : std::uint8_t {
    Direct,
    Shared,
};

enum class Bf16WeightCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Bf16PhaseOrder : std::uint8_t {
    Sequential,
    RowSwizzled,
};

enum class Bf16SimtActivationAccess : std::uint8_t {
    DirectStream,
    WarpPacked,
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane,
          int AccumulatorChains, Bf16ActivationAccess ActivationAccess, Bf16WeightCache WeightCache,
          Bf16PhaseOrder PhaseOrder, int PhaseStride, int PrefetchDepth, int PhaseUnroll,
          int MinBlocksPerSm>
struct Bf16A16GemvSchedule {
    static constexpr int kStaticK     = 0;
    static constexpr int kBlockTokens = 1;
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 4 || ValuesPerLane == 8 || ValuesPerLane == 16);
    static_assert(AccumulatorChains > 0 && AccumulatorChains <= ValuesPerLane);
    static_assert((AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(PrefetchDepth == 1 || PrefetchDepth == 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4 || PhaseUnroll == 8);
    static_assert(PhaseStride > 0);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kPhaseOrder       = PhaseOrder;
    static constexpr int kPhaseStride       = PhaseStride;
    static constexpr int kPrefetchDepth     = PrefetchDepth;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kBlockRows         = kRowGroupsPerCta * RowsPerWarp;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane,
          int AccumulatorChains, int TokenBatch, Bf16SimtActivationAccess ActivationAccess,
          Bf16WeightCache WeightCache, Bf16PhaseOrder PhaseOrder, int PhaseStride, int PhaseUnroll,
          int PrefetchDepth, int MinBlocksPerSm, int BlockTokens = 4>
struct Bf16A16SimtSchedule {
    static constexpr int kStaticK       = 0;
    static constexpr int kTokenCapacity = 0;
    static constexpr bool kExactTokens  = false;
    static constexpr int kBlockTokens   = BlockTokens;
    static_assert(BlockTokens > 0 && BlockTokens <= 32);
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 4 || ValuesPerLane == 8 || ValuesPerLane == 16);
    static_assert(AccumulatorChains > 0 && AccumulatorChains <= ValuesPerLane);
    static_assert((AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(TokenBatch == 1 || TokenBatch == 2 || TokenBatch == 4 || TokenBatch == 8);
    static_assert(PhaseStride > 0);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4 || PhaseUnroll == 8);
    static_assert(PrefetchDepth == 1 || PrefetchDepth == 2);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr int kTokenBatch        = TokenBatch;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kPhaseOrder       = PhaseOrder;
    static constexpr int kPhaseStride       = PhaseStride;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kPrefetchDepth     = PrefetchDepth;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kBlockRows         = kRowGroupsPerCta * RowsPerWarp;
};

enum class Bf16MmaFragmentPipeline : std::uint8_t {
    Serial,
    PingPong,
};

enum class Bf16MmaRaster : std::uint8_t {
    TokenFast,
    RowFast,
    Grouped,
};

enum class Bf16MmaSwizzle : std::uint8_t {
    Plain,
    Xor64,
    Tma128,
};

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Cache WeightCache, Cache ActivationCache,
          Bf16MmaFragmentPipeline FragmentPipeline, Bf16MmaRaster Raster,
          Bf16MmaSwizzle Swizzle = Bf16MmaSwizzle::Xor64, int RasterGroupRows = 1>
struct Bf16A16MmaSchedule {
    static constexpr int kStaticK                              = 0;
    static constexpr int kBlockRows                            = BlockRows;
    static constexpr int kBlockTokens                          = BlockTokens;
    static constexpr int kBlockK                               = BlockK;
    static constexpr int kWarpRows                             = WarpRows;
    static constexpr int kWarpTokens                           = WarpTokens;
    static constexpr int kStages                               = Stages;
    static constexpr int kMinBlocksPerSm                       = MinBlocksPerSm;
    static constexpr Cache kWeightCache                        = WeightCache;
    static constexpr Cache kActivationCache                    = ActivationCache;
    static constexpr Bf16MmaFragmentPipeline kFragmentPipeline = FragmentPipeline;
    static constexpr Bf16MmaRaster kRaster                     = Raster;
    static constexpr Bf16MmaSwizzle kSwizzle                   = Swizzle;
    static constexpr int kRasterGroupRows                      = RasterGroupRows;

    static constexpr int kWarpsRows      = kBlockRows / kWarpRows;
    static constexpr int kWarpsTokens    = kBlockTokens / kWarpTokens;
    static constexpr int kWarps          = kWarpsRows * kWarpsTokens;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kMmaRows        = kWarpRows / 16;
    static constexpr int kMmaTokens      = kWarpTokens / 8;
    static constexpr int kMmaK           = kBlockK / 16;
    static constexpr int kSharedElements = kStages * (kBlockRows + kBlockTokens) * kBlockK;
    static constexpr int kSharedBytes = kSharedElements * static_cast<int>(sizeof(__nv_bfloat16));

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert(kBlockRows % kWarpRows == 0 && kBlockTokens % kWarpTokens == 0);
    static_assert(kWarpRows % 16 == 0 && kWarpTokens % 8 == 0);
    static_assert(kBlockK % 64 == 0);
    static_assert(kStages >= 2 && kStages <= 8);
    static_assert(kMinBlocksPerSm >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 99 * 1024);
    static_assert(kRaster != Bf16MmaRaster::Grouped || kRasterGroupRows > 0);
};

template <int BlockRows, int BlockTokens, int KWarps, int WarpK = 64, int Stages = 2,
          Cache WeightCache = Cache::cg, Cache ActivationCache = Cache::cg, int MinBlocksPerSm = 1>
struct Bf16A16SlicedKMmaSchedule {
    static_assert(BlockRows > 0 && BlockRows % 16 == 0);
    static_assert(BlockTokens > 0 && BlockTokens % 8 == 0);
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(WarpK > 0 && WarpK % 64 == 0);
    static_assert(Stages >= 1 && Stages <= 8 && MinBlocksPerSm > 0);
    static constexpr int kStaticK            = 0;
    static constexpr int kBlockRows          = BlockRows;
    static constexpr int kBlockTokens        = BlockTokens;
    static constexpr int kKWarps             = KWarps;
    static constexpr int kWarpK              = WarpK;
    static constexpr int kBlockK             = KWarps * WarpK;
    static constexpr int kStages             = Stages;
    static constexpr int kThreads            = KWarps * 32;
    static constexpr int kMinBlocksPerSm     = MinBlocksPerSm;
    static constexpr int kMmaRows            = BlockRows / 16;
    static constexpr int kMmaTokens          = BlockTokens / 8;
    static constexpr Cache kWeightCache      = WeightCache;
    static constexpr Cache kActivationCache  = ActivationCache;
    static constexpr Bf16MmaSwizzle kSwizzle = Bf16MmaSwizzle::Xor64;
    static constexpr int kStageElements      = (BlockRows + BlockTokens) * kBlockK;
    static constexpr int kStagingBytes       = Stages * kStageElements * 2;
    static constexpr int kPartialBytes       = KWarps * BlockRows * BlockTokens * sizeof(float);
    static constexpr int kSharedBytes =
        kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm = 1, Bf16MmaRaster Raster = Bf16MmaRaster::TokenFast,
          int RasterGroupRows = 1>
struct Bf16A16TmaMmaSchedule
    : Bf16A16MmaSchedule<BlockRows, BlockTokens, BlockK, WarpRows, WarpTokens, Stages,
                         MinBlocksPerSm, Cache::cg, Cache::cg, Bf16MmaFragmentPipeline::PingPong,
                         Raster, Bf16MmaSwizzle::Tma128, RasterGroupRows> {
    static constexpr int kProducerThreads = 32;
    static constexpr int kConsumerWarps   = (BlockRows / WarpRows) * (BlockTokens / WarpTokens);
    static constexpr int kThreads         = kProducerThreads + kConsumerWarps * 32;
    static constexpr int kTensorBytes     = Stages * (BlockRows + BlockTokens) * BlockK * 2;
    static constexpr int kBarrierBytes    = Stages * 2 * sizeof(std::uint64_t);
    static constexpr int kSharedBytes     = kTensorBytes + kBarrierBytes;
    static_assert(kThreads <= 1024 && kSharedBytes <= 99 * 1024);
    static_assert(BlockRows <= 256 && BlockTokens <= 256 && BlockK <= 16384);
};

// Static K and optional whole-call token specialization do not restrict the generic template.
template <class Schedule, int K, int Capacity = 0, bool ExactTokens = false>
struct Bf16ScheduleInstance : Schedule {
    static_assert(K > 0 && K % 8 == 0);
    static_assert(Capacity >= 0 && Capacity <= 32);
    static_assert(!ExactTokens || Capacity > 0);
    static_assert(
        Capacity == 0 || requires { Schedule::kTokenCapacity; },
        "whole-call token capacity is a SIMT specialization");
    static constexpr int kStaticK       = K;
    static constexpr int kTokenCapacity = Capacity;
    static constexpr bool kExactTokens  = ExactTokens;
    static constexpr int kBlockTokens   = Capacity ? Capacity : Schedule::kBlockTokens;
};
} // namespace ninfer::ops::detail
