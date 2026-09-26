#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

namespace ninfer::ops::detail {

template <int BlockRows, int BlockTokens, int WarpsPerRow, int GroupsPerWarpStage, int Stages,
          Cache CodeCache, int MinBlocksPerSm, bool Predicated = false>
struct Q4A16SimtSchedule {
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
    static constexpr int kStageK             = GroupsPerWarpStage * Q4RowSplitStorage::kGroupK;
    static constexpr int kCodeVecsPerStage   = GroupsPerWarpStage * 2;
    static constexpr int kScalePairsPerStage = GroupsPerWarpStage / 2;
    static constexpr int kCodePhases         = (GroupsPerWarpStage + 3) / 4;
    static constexpr int kStagingBytes =
        kWarps * Stages * GroupsPerWarpStage *
        (Q4RowSplitStorage::kCodeBytesPerGroup + Q4RowSplitStorage::kScaleBytesPerGroup);
    static constexpr int kPartialBytes = WarpsPerRow > 1 ? kWarps* BlockTokens * sizeof(float) : 0;
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024, "Q4 SIMT shared memory exceeds 48 KiB");
};

enum class Q4MmaFragmentPipeline {
    Serial,
    PingPong,
};

enum class Q4ScaleLoad {
    Scalar16,
    Pair32,
};

template <int BlockRows, int BlockTokens, int BlockK, int WarpRows, int WarpTokens, int Stages,
          int MinBlocksPerSm, Q4MmaFragmentPipeline FragmentPipeline, Cache WeightCache,
          Cache ActivationCache, Q4ScaleLoad ScaleLoadMode, int ActivationStages = Stages>
struct Q4A16MmaSchedule {
    static constexpr int kBlockRows   = BlockRows;
    static constexpr int kBlockTokens = BlockTokens;
    static constexpr int kBlockK      = BlockK;
    static constexpr int kWarpRows    = WarpRows;
    static constexpr int kWarpTokens  = WarpTokens;

    static constexpr int kStages                             = Stages;
    static constexpr int kMinBlocksPerSm                     = MinBlocksPerSm;
    static constexpr Q4MmaFragmentPipeline kFragmentPipeline = FragmentPipeline;
    static constexpr Cache kWeightCache                      = WeightCache;
    static constexpr Cache kActivationCache                  = ActivationCache;
    static constexpr Q4ScaleLoad kScaleLoadMode              = ScaleLoadMode;
    static constexpr int kActivationStages                   = ActivationStages;

    static constexpr int kWarpGridRows   = kBlockRows / kWarpRows;
    static constexpr int kWarpGridTokens = kBlockTokens / kWarpTokens;
    static constexpr int kWarps          = kWarpGridRows * kWarpGridTokens;
    static constexpr int kThreads        = kWarps * 32;
    static constexpr int kMmaRows        = kWarpRows / 16;
    static constexpr int kMmaTokens      = kWarpTokens / 8;
    static constexpr int kMmaKSteps      = kBlockK / 16;
    static constexpr int kGroupsPerK     = kBlockK / Q4RowSplitStorage::kGroupK;
    static constexpr int kScaleBytes =
        kScaleLoadMode == Q4ScaleLoad::Pair32 ? 4 : Q4RowSplitStorage::kScaleBytesPerGroup;

    static constexpr int kSharedBytes =
        kBlockRows * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kActivationStages * kBlockTokens * kBlockK * static_cast<int>(sizeof(__nv_bfloat16)) +
        kStages * kBlockRows * kGroupsPerK * Q4RowSplitStorage::kCodeBytesPerGroup +
        kStages * kBlockRows * kGroupsPerK * kScaleBytes;

    static_assert(kBlockRows > 0 && kBlockTokens > 0 && kBlockK > 0);
    static_assert(kBlockK % Q4RowSplitStorage::kGroupK == 0,
                  "Q4 MMA K tile must contain complete quant groups");
    static_assert(kBlockRows % kWarpRows == 0 && kBlockTokens % kWarpTokens == 0,
                  "Q4 MMA block tile must divide into warp tiles");
    static_assert(kWarpRows % 16 == 0 && kWarpTokens % 8 == 0,
                  "Q4 MMA warp tile must be composed of m16n8 MMA tiles");
    static_assert(kStages >= 1 && kStages <= 8, "Q4 MMA cp.async pipeline depth must fit cp_wait");
    static_assert(kActivationStages == 1 || kActivationStages == kStages,
                  "Q4 MMA activation staging is single-buffered or follows the quant pipeline");
    static_assert(kActivationStages == kStages || kStages < 8,
                  "Q4 single-buffered activations need a separate async group");
    static_assert(kMinBlocksPerSm >= 1);
    static_assert(kWarps >= 1 && kThreads <= 1024);
    static_assert(kSharedBytes <= 48 * 1024,
                  "Q4 MMA staged shared memory exceeds the static 48 KiB budget");
};

enum class Q4SlicedKReduction {
    Linear,
    Pairwise,
};

template <int BlockRows, int BlockTokens, int KWarps, int Stages, Cache WeightCache,
          Cache ActivationCache, int MinBlocksPerSm, int StaticK = 0,
          int TokenCapacity            = BlockTokens,
          Q4SlicedKReduction Reduction = Q4SlicedKReduction::Linear>
struct Q4A16SlicedKMmaSchedule {
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
    static constexpr int kStagingBytes =
        Stages *
        (BlockRows * KWarps *
             (Q4RowSplitStorage::kCodeBytesPerGroup + Q4RowSplitStorage::kScaleBytesPerGroup) +
         KWarps * BlockTokens * 64 * 2);
    static constexpr int kPartialBytes = KWarps * BlockRows * BlockTokens * sizeof(float);
    static constexpr int kSharedBytes =
        ((kStagingBytes > kPartialBytes ? kStagingBytes : kPartialBytes) + 15) / 16 * 16;
    static_assert(kSharedBytes <= 48 * 1024, "Q4 sliced-K shared memory exceeds 48 KiB");
};

enum class Q4GemvActivationAccess {
    Direct,
    CtaSharedFullK,
};

enum class Q4GemvLaneMapping {
    PackedByte2,
    PackedWord8,
};

enum class Q4GemvDecodeMode {
    ScalarInteger,
    Fp16Mantissa,
};

enum class Q4GemvCodeTransfer {
    SyncVector16,
    AsyncVector16,
};

enum class Q4GemvScaleAccess {
    Scalar16Shuffle,
    SharedPair32,
};

template <int RowsPerCta, int WarpsPerRow, int GroupsPerWarpTile, int PipelineStages,
          Q4GemvActivationAccess ActivationAccess, Q4GemvLaneMapping LaneMapping,
          Q4GemvDecodeMode DecodeMode, Q4GemvCodeTransfer CodeTransfer,
          Q4GemvScaleAccess ScaleAccess, Cache CodeCache, int StaticGroupsPerRow,
          int LaunchBoundsMinBlocks>
struct Q4A16GemvSchedule {
    static_assert(RowsPerCta > 0, "Q4 GEMV requires at least one row per CTA");
    static_assert(WarpsPerRow > 0, "Q4 GEMV requires at least one warp per row");
    static_assert(GroupsPerWarpTile > 0 && (GroupsPerWarpTile % 2) == 0,
                  "Q4 GEMV group tiles must contain whole scale pairs");
    static_assert(GroupsPerWarpTile <= 32,
                  "Q4 GEMV scalar-shuffle scale access is limited to one warp");
    static_assert(PipelineStages >= 1 && PipelineStages <= 8,
                  "Q4 GEMV pipeline depth must fit cp.async wait-group immediates");
    static_assert(StaticGroupsPerRow == 0 ||
                      (StaticGroupsPerRow > 0 && (StaticGroupsPerRow % 2) == 0),
                  "Q4 GEMV static row ownership must contain whole scale pairs");
    static_assert(LaunchBoundsMinBlocks >= 1, "Q4 GEMV launch-bounds occupancy must be positive");

    static constexpr int kBlockRows          = RowsPerCta;
    static constexpr int kWarpsPerRow        = WarpsPerRow;
    static constexpr int kGroupsPerWarpTile  = GroupsPerWarpTile;
    static constexpr int kStages             = PipelineStages;
    static constexpr auto kActivationAccess  = ActivationAccess;
    static constexpr auto kLaneMapping       = LaneMapping;
    static constexpr auto kDecodeMode        = DecodeMode;
    static constexpr auto kCodeTransfer      = CodeTransfer;
    static constexpr auto kScaleAccess       = ScaleAccess;
    static constexpr auto kCodeCache         = CodeCache;
    static constexpr int kStaticGroupsPerRow = StaticGroupsPerRow;
    static constexpr int kMinBlocksPerSm     = LaunchBoundsMinBlocks;

    static constexpr int kWarps   = kBlockRows * kWarpsPerRow;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kCodeVectorsPerTile =
        kGroupsPerWarpTile * Q4RowSplitStorage::kCodeBytesPerGroup / sizeof(uint4);
    static constexpr int kScalePairsPerTile = kGroupsPerWarpTile / 2;

    static constexpr int kBlockTokens = 1;
    static constexpr int kStagingBytes =
        kWarps * kStages *
        (kGroupsPerWarpTile * Q4RowSplitStorage::kCodeBytesPerGroup +
         (kScaleAccess == Q4GemvScaleAccess::SharedPair32 ? kScalePairsPerTile * 4 : 0));
    static constexpr int kSharedBytes = (kStagingBytes + kWarps * 4 + 15) / 16 * 16;
    static_assert(kSharedBytes <= 48 * 1024);
    static_assert(kWarps <= 32, "Q4 GEMV cannot exceed the CUDA CTA warp limit");
    static_assert(kThreads <= 1024, "Q4 GEMV cannot exceed the CUDA CTA thread limit");
    static_assert((kGroupsPerWarpTile * Q4RowSplitStorage::kCodeBytesPerGroup) % sizeof(uint4) == 0,
                  "Q4 GEMV code tile must be representable as 16-byte vectors");

    static_assert((kLaneMapping == Q4GemvLaneMapping::PackedByte2 &&
                   kDecodeMode == Q4GemvDecodeMode::ScalarInteger) ||
                      (kLaneMapping == Q4GemvLaneMapping::PackedWord8 &&
                       kDecodeMode == Q4GemvDecodeMode::Fp16Mantissa),
                  "Q4 GEMV lane mapping and decode mode must describe the same packed ownership");
    static_assert((kCodeTransfer == Q4GemvCodeTransfer::SyncVector16 &&
                   kScaleAccess == Q4GemvScaleAccess::Scalar16Shuffle && kStages == 1 &&
                   kCodeCache == Cache::ca) ||
                      (kCodeTransfer == Q4GemvCodeTransfer::AsyncVector16 &&
                       kScaleAccess == Q4GemvScaleAccess::SharedPair32),
                  "Q4 GEMV transfer and scale schedules must select an implemented path");
};

} // namespace ninfer::ops::detail
