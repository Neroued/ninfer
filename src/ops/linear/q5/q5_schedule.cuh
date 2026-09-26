#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

namespace ninfer::ops::detail {

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage, int Stages,
          Cache CodeCache, int MinBlocksPerSm, bool Predicated = false>
struct Q5A16SimtSchedule {
    static_assert(BlockRows > 0 && BlockRows <= 32);
    static_assert(BlockTokens > 0 && BlockTokens <= 8);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(GroupsPerWarpStage > 0 && GroupsPerWarpStage % 2 == 0);
    static_assert(Stages >= 2 && Stages <= 8);
    static_assert(MinBlocksPerSm > 0);
    static constexpr bool kPredicated        = Predicated;
    static constexpr int kBlockRows          = BlockRows;
    static constexpr int kBlockTokens        = BlockTokens;
    static constexpr int kWarpsPerRow        = WarpsPerRow;
    static constexpr int kGroupsPerWarpStage = GroupsPerWarpStage;
    static constexpr int kStages             = Stages;
    static constexpr Cache kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm     = MinBlocksPerSm;
    static constexpr int kWarps              = BlockRows * WarpsPerRow;
    static constexpr int kThreads            = kWarps * 32;
    static constexpr int kStageK             = GroupsPerWarpStage * Q5RowSplitStorage::kGroupK;
    static constexpr int kCodeVecsPerStage   = GroupsPerWarpStage * 2;
    static constexpr int kHighVecsPerStage   = GroupsPerWarpStage / 2;
    static constexpr int kScalePairsPerStage = GroupsPerWarpStage / 2;
    static constexpr int kCodePhases         = (GroupsPerWarpStage + 3) / 4;
    static constexpr int kStagingBytes =
        kWarps * Stages * GroupsPerWarpStage *
        (Q5RowSplitStorage::kCodeBytesPerGroup + Q5RowSplitStorage::kHighBytesPerGroup +
         Q5RowSplitStorage::kScaleBytesPerGroup);
    static constexpr int kPartialBytes = WarpsPerRow > 1 ? kWarps* BlockTokens * sizeof(float) : 0;
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024, "Q5 SIMT shared memory exceeds 48 KiB");
};

enum class Q5MmaFragmentPipeline {
    Serial,
    PingPong,
};

enum class Q5ScaleLoad {
    Scalar16,
    Pair32,
};

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Q5MmaFragmentPipeline FragmentPipeline, Cache WeightCache,
          Cache ActivationCache, Q5ScaleLoad ScaleLoadMode, int ActivationStages = Stages>
struct Q5A16MmaSchedule {
    static constexpr int kBlockRows   = BlockRows;
    static constexpr int kBlockTokens = BlockTokens;
    static constexpr int kBlockK      = BlockK;
    static constexpr int kWarpRows    = WarpRows;
    static constexpr int kWarpTokens  = WarpTokens;

    static constexpr int kStages                             = Stages;
    static constexpr int kMinBlocksPerSm                     = MinBlocksPerSm;
    static constexpr Q5MmaFragmentPipeline kFragmentPipeline = FragmentPipeline;
    static constexpr Cache kWeightCache                      = WeightCache;
    static constexpr Cache kActivationCache                  = ActivationCache;
    static constexpr Q5ScaleLoad kScaleLoadMode              = ScaleLoadMode;
    static constexpr int kActivationStages                   = ActivationStages;

    static constexpr int kWarpGridRows   = kBlockRows / kWarpRows;
    static constexpr int kWarpGridTokens = kBlockTokens / kWarpTokens;
    static constexpr int kWarps          = kWarpGridRows * kWarpGridTokens;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kMmaRows        = kWarpRows / 16;
    static constexpr int kMmaTokens      = kWarpTokens / 8;
    static constexpr int kMmaKSteps      = kBlockK / 16;
    static constexpr int kGroupsPerK     = kBlockK / Q5RowSplitStorage::kGroupK;
    static constexpr int kScaleBytes =
        kScaleLoadMode == Q5ScaleLoad::Pair32 ? 4 : Q5RowSplitStorage::kScaleBytesPerGroup;

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kStages * kBlockRows * kGroupsPerK * Q5RowSplitStorage::kCodeBytesPerGroup +
        kStages * kBlockRows * kGroupsPerK * (kScaleBytes + Q5RowSplitStorage::kHighBytesPerGroup);

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert(kBlockK % Q5RowSplitStorage::kGroupK == 0,
                  "Q5 MMA K tile must contain complete quant groups");
    static_assert(kBlockRows % kWarpRows == 0 && kBlockTokens % kWarpTokens == 0,
                  "Q5 MMA block tile must divide into warp tiles");
    static_assert(kWarpRows % 16 == 0 && kWarpTokens % 8 == 0,
                  "Q5 MMA warp tile must be composed of m16n8 MMA tiles");
    static_assert(kStages >= 1 && kStages <= 8, "Q5 MMA cp.async pipeline depth must fit cp_wait");
    static_assert(kActivationStages == 1 || kActivationStages == kStages,
                  "Q5 MMA activation staging is single-buffered or follows the quant pipeline");
    static_assert(kActivationStages == kStages || kStages < 8,
                  "Q5 single-buffered activations need a separate async group");
    static_assert(kMinBlocksPerSm >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024,
                  "Q5 MMA staged shared memory exceeds the static 48 KiB budget");
};

enum class Q5SlicedKReduction {
    Linear,
    Pairwise,
};

template <int BlockRows, int BlockTokens, int KWarps, int Stages, Cache WeightCache,
          Cache ActivationCache, int MinBlocksPerSm, int StaticK = 0,
          int TokenCapacity            = BlockTokens,
          Q5SlicedKReduction Reduction = Q5SlicedKReduction::Linear>
struct Q5A16SlicedKMmaSchedule {
    static_assert(BlockRows == 16 || BlockRows == 32);
    static_assert(BlockTokens >= 8 && BlockTokens <= 64 && BlockTokens % 8 == 0);
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
    static constexpr int kBlockRows   = BlockRows;
    static constexpr int kBlockTokens = BlockTokens;
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % 128 == 0));
    static_assert(TokenCapacity > 0 && TokenCapacity <= BlockTokens);
    // Static K specializes a padding-free geometry, including physical row strides.
    static constexpr auto kReduction        = Reduction;
    static constexpr int kStaticK           = StaticK;
    static constexpr int kTokenCapacity     = TokenCapacity;
    static constexpr int kKWarps            = KWarps;
    static constexpr int kWarps             = KWarps;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kWarpK             = 64;
    static constexpr int kBlockK            = KWarps * kWarpK;
    static constexpr int kStages            = Stages;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr Cache kActivationCache = ActivationCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kMmaRows           = BlockRows / 16;
    static constexpr int kMmaTokens         = BlockTokens / 8;
    // Code/scale planes and BF16 activations. Partial results reuse this storage.
    static constexpr int kStagingBytes = Stages * (BlockRows * KWarps *
                                                       (Q5RowSplitStorage::kCodeBytesPerGroup +
                                                        Q5RowSplitStorage::kHighBytesPerGroup +
                                                        Q5RowSplitStorage::kScaleBytesPerGroup) +
                                                   KWarps * BlockTokens * 64 * 2);
    static constexpr int kPartialBytes = KWarps * BlockRows * BlockTokens * sizeof(float);
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kSharedBytes <= 48 * 1024, "Q5 sliced-K shared memory exceeds 48 KiB");
};

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpTile,
          int MinBlocksPerSm, int StaticK = 0, bool ExactTokens = false>
struct Q5A16DirectSimtSchedule {
    static_assert(BlockRows > 0 && BlockTokens > 0 && BlockTokens <= 32);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(GroupsPerWarpTile > 0 && GroupsPerWarpTile % 4 == 0);
    static_assert(MinBlocksPerSm > 0 && (StaticK == 0 || (StaticK > 0 && StaticK % 8 == 0)));
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kGroupsPerWarpTile = GroupsPerWarpTile;
    static constexpr int kPhasesPerWarp     = GroupsPerWarpTile / 4;
    static constexpr int kBlockK            = WarpsPerRow * GroupsPerWarpTile * 64;
    static constexpr bool kExactTokens      = ExactTokens;
    static constexpr int kStaticK           = StaticK;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kWarps             = BlockRows * WarpsPerRow;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kSharedBytes       = kWarps * BlockTokens * sizeof(float);
    static_assert(kThreads <= 1024 && kSharedBytes <= 48 * 1024);
};

template <int BlockRows, int WarpsPerRow, int GroupsPerWarpTile, int Stages, Cache WeightCache,
          int MinBlocksPerSm, bool StageX = false, int StaticK = 0>
struct Q5A16GemvSchedule {
    static_assert(BlockRows > 0 && WarpsPerRow > 0 && BlockRows * WarpsPerRow <= 32);
    static_assert(GroupsPerWarpTile > 0 && GroupsPerWarpTile % 2 == 0);
    static_assert(Stages >= 1 && Stages <= 8 && MinBlocksPerSm > 0);
    static_assert(StaticK == 0 || (StaticK > 0 && StaticK % 128 == 0));
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = 1;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kGroupsPerWarpTile = GroupsPerWarpTile;
    static constexpr int kStages            = Stages;
    static constexpr Cache kWeightCache     = WeightCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr bool kStageX           = StageX;
    static constexpr int kStaticK           = StaticK;
    static constexpr int kWarps             = BlockRows * WarpsPerRow;
    static constexpr int kThreads           = kWarps * 32;
    static constexpr int kSharedBytes =
        (kWarps * Stages * GroupsPerWarpTile * 42 + kWarps * 4 + 15) / 16 * 16;
    static_assert(kSharedBytes <= 48 * 1024);
};

} // namespace ninfer::ops::detail
