#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/q6/q6_rowsplit_storage.cuh"

namespace ninfer::ops::detail {

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage, int Stages,
          Cache CodeCache, int MinBlocksPerSm>
struct Q6A16SimtSchedule {
    static_assert(BlockRows > 0 && BlockRows <= 32);
    static_assert(BlockTokens > 0 && BlockTokens <= 8);
    static_assert(WarpsPerRow == 1 || WarpsPerRow == 2 || WarpsPerRow == 4 || WarpsPerRow == 8);
    static_assert(GroupsPerWarpStage > 0 && GroupsPerWarpStage % 2 == 0);
    static_assert(Stages >= 2 && Stages <= 8);
    static_assert(MinBlocksPerSm > 0);
    static constexpr int kBlockRows          = BlockRows;
    static constexpr int kBlockTokens        = BlockTokens;
    static constexpr int kWarpsPerRow        = WarpsPerRow;
    static constexpr int kGroupsPerWarpStage = GroupsPerWarpStage;
    static constexpr int kStages             = Stages;
    static constexpr Cache kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm     = MinBlocksPerSm;
    static constexpr int kWarps              = BlockRows * WarpsPerRow;
    static constexpr int kThreads            = kWarps * 32;
    static constexpr int kStageK             = GroupsPerWarpStage * Q6RowSplitStorage::kGroupK;
    static constexpr int kCodeVecsPerStage   = GroupsPerWarpStage * 2;
    static constexpr int kHighVecsPerStage   = GroupsPerWarpStage;
    static constexpr int kScalePairsPerStage = GroupsPerWarpStage / 2;
    static constexpr int kCodePhases         = (GroupsPerWarpStage + 3) / 4;
    static constexpr int kStagingBytes =
        kWarps * Stages * GroupsPerWarpStage *
        (Q6RowSplitStorage::kCodeBytesPerGroup + Q6RowSplitStorage::kHighBytesPerGroup +
         Q6RowSplitStorage::kScaleBytesPerGroup);
    static constexpr int kPartialBytes = WarpsPerRow > 1 ? kWarps* BlockTokens * sizeof(float) : 0;
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024, "Q6 SIMT shared memory exceeds 48 KiB");
};

template <int BlockRows, int WarpsPerRow, int GroupsPerWarpStage, int Stages, Cache CodeCache,
          int MinBlocksPerSm>
using Q6A16GemvSchedule = Q6A16SimtSchedule<BlockRows, 1, WarpsPerRow, GroupsPerWarpStage, Stages,
                                            CodeCache, MinBlocksPerSm>;

enum class Q6MmaFragmentPipeline {
    Serial,
    PingPong,
};

enum class Q6ScaleLoad {
    Scalar16,
    Pair32,
};

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Q6MmaFragmentPipeline FragmentPipeline, Cache WeightCache,
          Cache ActivationCache, Q6ScaleLoad ScaleLoadMode, int ActivationStages = Stages>
struct Q6A16MmaSchedule {
    static constexpr int kBlockRows   = BlockRows;
    static constexpr int kBlockTokens = BlockTokens;
    static constexpr int kBlockK      = BlockK;
    static constexpr int kWarpRows    = WarpRows;
    static constexpr int kWarpTokens  = WarpTokens;

    static constexpr int kStages                             = Stages;
    static constexpr int kMinBlocksPerSm                     = MinBlocksPerSm;
    static constexpr Q6MmaFragmentPipeline kFragmentPipeline = FragmentPipeline;
    static constexpr Cache kWeightCache                      = WeightCache;
    static constexpr Cache kActivationCache                  = ActivationCache;
    static constexpr Q6ScaleLoad kScaleLoadMode              = ScaleLoadMode;
    static constexpr int kActivationStages                   = ActivationStages;

    static constexpr int kWarpGridRows   = kBlockRows / kWarpRows;
    static constexpr int kWarpGridTokens = kBlockTokens / kWarpTokens;
    static constexpr int kWarps          = kWarpGridRows * kWarpGridTokens;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kMmaRows        = kWarpRows / 16;
    static constexpr int kMmaTokens      = kWarpTokens / 8;
    static constexpr int kMmaKSteps      = kBlockK / 16;
    static constexpr int kGroupsPerK     = kBlockK / Q6RowSplitStorage::kGroupK;
    static constexpr int kScaleBytes =
        kScaleLoadMode == Q6ScaleLoad::Pair32 ? 4 : Q6RowSplitStorage::kScaleBytesPerGroup;

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kStages * kBlockRows * kGroupsPerK * Q6RowSplitStorage::kCodeBytesPerGroup +
        kStages * kBlockRows * kGroupsPerK * Q6RowSplitStorage::kHighBytesPerGroup +
        kStages * kBlockRows * kGroupsPerK * kScaleBytes;

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert(kBlockK % Q6RowSplitStorage::kGroupK == 0,
                  "Q6 MMA K tile must contain complete quant groups");
    static_assert(kBlockRows % kWarpRows == 0 && kBlockTokens % kWarpTokens == 0,
                  "Q6 MMA block tile must divide into warp tiles");
    static_assert(kWarpRows % 16 == 0 && kWarpTokens % 8 == 0,
                  "Q6 MMA warp tile must be composed of m16n8 MMA tiles");
    static_assert(kStages >= 1 && kStages <= 8, "Q6 MMA cp.async pipeline depth must fit cp_wait");
    static_assert(kActivationStages == 1 || kActivationStages == kStages,
                  "Q6 MMA activation staging is single-buffered or follows the quant pipeline");
    static_assert(kMinBlocksPerSm >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024,
                  "Q6 MMA staged shared memory exceeds the static 48 KiB budget");
};

template <int BlockRows, int BlockTokens, int KWarps, int Stages, Cache WeightCache,
          Cache ActivationCache, int MinBlocksPerSm>
struct Q6A16SlicedKMmaSchedule {
    static_assert(BlockRows == 16 || BlockRows == 32);
    static_assert(BlockTokens >= 8 && BlockTokens <= 64 && BlockTokens % 8 == 0);
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8);
    static_assert(Stages == 1 || Stages == 2);
    static_assert(MinBlocksPerSm > 0);
    static constexpr int kBlockRows         = BlockRows;
    static constexpr int kBlockTokens       = BlockTokens;
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
    // Code/high/scale planes and BF16 activations. Partial results reuse this storage.
    static constexpr int kStagingBytes = Stages * (BlockRows * KWarps *
                                                       (Q6RowSplitStorage::kCodeBytesPerGroup +
                                                        Q6RowSplitStorage::kHighBytesPerGroup +
                                                        Q6RowSplitStorage::kScaleBytesPerGroup) +
                                                   KWarps * BlockTokens * 64 * 2);
    static constexpr int kPartialBytes = KWarps * BlockRows * BlockTokens * sizeof(float);
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kSharedBytes <= 48 * 1024, "Q6 sliced-K shared memory exceeds 48 KiB");
};

} // namespace ninfer::ops::detail
