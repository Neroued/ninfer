#pragma once

#include "ops/common/memory.cuh"

#include <cstdint>

namespace ninfer::ops::detail {

enum class W8SmallTMmaScaleAccess : std::uint8_t {
    Direct,
    Shared,
};

enum class W8SmallTMmaActivationStage : std::uint8_t {
    ActiveOnly,
    PaddedZero,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct W8LinearGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 16) == 0);
    static_assert((InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows    = OutputRows;
    static constexpr std::int32_t kInputRows     = InputRows;
    static constexpr std::int32_t kGroupsPerRow  = InputRows / 32;
    static constexpr std::int32_t kCodeRowBytes  = InputRows;
    static constexpr std::int32_t kScaleRowBytes = kGroupsPerRow * sizeof(std::uint16_t);
};

template <int KWarps, int TileTokens, int MinBlocksPerSm, W8SmallTMmaScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          W8SmallTMmaActivationStage ActivationStage = W8SmallTMmaActivationStage::ActiveOnly>
struct W8SmallTMmaSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48 || TileTokens == 56 || TileTokens == 64 ||
                  TileTokens == 72 || TileTokens == 80 || TileTokens == 88);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = KWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / KWarps;
    static constexpr int kScaleBytesPerRow  = kGroupK / 16;
};

template <int TileTokens, int ActiveTokens>
using W8SmallTMmaDefaultSchedule = W8SmallTMmaSchedule<
    8, TileTokens, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
    (ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct)>;

using W8VocabularyProjectionGeometry       = W8LinearGeometry<248320, 5120>;
using W8MtpInputProjectionGeometry         = W8LinearGeometry<5120, 10240>;
using W8MtpAttentionProjectionGeometry     = W8LinearGeometry<14336, 5120>;
using W8MtpAttentionOutputGeometry         = W8LinearGeometry<5120, 6144>;
using W8MtpGateUpProjectionGeometry        = W8LinearGeometry<34816, 5120>;
using W8MtpDownProjectionGeometry          = W8LinearGeometry<5120, 17408>;
using W835bMtpProjectionGeometry           = W8LinearGeometry<2048, 4096>;
using W8DFlash2AttentionProjectionGeometry = W8LinearGeometry<6144, 5120>;
using W8N5120K25600Geometry                = W8LinearGeometry<5120, 25600>;

inline constexpr std::int32_t kW8VocabularyFirstSmallT         = 1;
inline constexpr std::int32_t kW8VocabularyLastSmallT          = 33;
inline constexpr std::int32_t kW8MtpInputFirstSmallT           = 1;
inline constexpr std::int32_t kW8MtpInputLastSmallT            = 48;
inline constexpr std::int32_t kW8MtpAttentionFirstSmallT       = 1;
inline constexpr std::int32_t kW8MtpAttentionLastSmallT        = 48;
inline constexpr std::int32_t kW8MtpAttentionOutputFirstSmallT = 1;
inline constexpr std::int32_t kW8MtpAttentionOutputLastSmallT  = 48;
inline constexpr std::int32_t kW8MtpGateUpFirstSmallT          = 1;
inline constexpr std::int32_t kW8MtpGateUpLastSmallT           = 52;
inline constexpr std::int32_t kW8MtpDownFirstSmallT            = 1;
inline constexpr std::int32_t kW8MtpDownLastSmallT             = 48;
inline constexpr std::int32_t kW835bMtpProjectionFirstSmallT   = 1;
inline constexpr std::int32_t kW835bMtpProjectionLastSmallT    = 48;
inline constexpr std::int32_t kW8DFlash2AttentionFirstSmallT   = 1;
inline constexpr std::int32_t kW8DFlash2AttentionLastSmallT    = 53;

template <class Geometry, int ActiveTokens>
struct W8LinearSmallTProductionSchedule;

template <int ActiveTokens>
struct W8LinearSmallTProductionSchedule<W8MtpInputProjectionGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kW8MtpInputFirstSmallT);
    static_assert(ActiveTokens <= kW8MtpInputLastSmallT);

    static constexpr int kTileTokens = ActiveTokens <= 8    ? 8
                                       : ActiveTokens <= 16 ? 16
                                       : ActiveTokens <= 24 ? 24
                                       : ActiveTokens <= 32 ? 32
                                       : ActiveTokens <= 40 ? 40
                                                            : 48;
    static constexpr int kKWarps     = ActiveTokens <= 24 ? 8 : 4;
    static constexpr auto kScaleAccess =
        ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    using Type = W8SmallTMmaSchedule<kKWarps, kTileTokens, 2, kScaleAccess>;
};

template <int ActiveTokens>
struct W8LinearSmallTProductionSchedule<W8MtpAttentionProjectionGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kW8MtpAttentionFirstSmallT);
    static_assert(ActiveTokens <= kW8MtpAttentionLastSmallT);

    static constexpr int kTileTokens = ActiveTokens <= 8    ? 8
                                       : ActiveTokens <= 16 ? 16
                                       : ActiveTokens <= 24 ? 24
                                       : ActiveTokens <= 32 ? 32
                                       : ActiveTokens <= 40 ? 40
                                                            : 48;
    static constexpr int kKWarps =
        ActiveTokens <= 4 || (ActiveTokens >= 17 && ActiveTokens <= 22) ? 8 : 4;
    static constexpr auto kScaleAccess =
        ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    using Type = W8SmallTMmaSchedule<kKWarps, kTileTokens, 3, kScaleAccess>;
};

template <int ActiveTokens>
struct W8LinearSmallTProductionSchedule<W8MtpAttentionOutputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kW8MtpAttentionOutputFirstSmallT);
    static_assert(ActiveTokens <= kW8MtpAttentionOutputLastSmallT);

    static constexpr int kTileTokens = ActiveTokens <= 8    ? 8
                                       : ActiveTokens <= 16 ? 16
                                       : ActiveTokens <= 24 ? 24
                                       : ActiveTokens <= 32 ? 32
                                       : ActiveTokens <= 40 ? 40
                                                            : 48;
    static constexpr int kKWarps     = ActiveTokens <= 31 ? 8 : 4;
    static constexpr auto kScaleAccess =
        ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    using Type = W8SmallTMmaSchedule<kKWarps, kTileTokens, 2, kScaleAccess, Cache::ca, Cache::cg,
                                     W8SmallTMmaActivationStage::PaddedZero>;
};

template <int ActiveTokens>
struct W8LinearSmallTProductionSchedule<W8MtpGateUpProjectionGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kW8MtpGateUpFirstSmallT);
    static_assert(ActiveTokens <= kW8MtpGateUpLastSmallT);

    static constexpr int kTileTokens = ActiveTokens <= 8    ? 8
                                       : ActiveTokens <= 16 ? 16
                                       : ActiveTokens <= 24 ? 24
                                       : ActiveTokens <= 32 ? 32
                                       : ActiveTokens <= 40 ? 40
                                       : ActiveTokens <= 48 ? 48
                                                            : 56;
    static constexpr int kKWarps     = ActiveTokens >= 22 && ActiveTokens <= 24 ? 8 : 4;
    static constexpr auto kScaleAccess =
        ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    static constexpr auto kActivationStage =
        ActiveTokens <= 4 || (ActiveTokens >= 9 && ActiveTokens <= 15)
            ? W8SmallTMmaActivationStage::PaddedZero
            : W8SmallTMmaActivationStage::ActiveOnly;
    using Type = W8SmallTMmaSchedule<kKWarps, kTileTokens, 2, kScaleAccess, Cache::ca, Cache::cg,
                                     kActivationStage>;
};

template <int ActiveTokens>
struct W8LinearSmallTProductionSchedule<W8MtpDownProjectionGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kW8MtpDownFirstSmallT);
    static_assert(ActiveTokens <= kW8MtpDownLastSmallT);

    static constexpr int kTileTokens = ActiveTokens <= 8    ? 8
                                       : ActiveTokens <= 16 ? 16
                                       : ActiveTokens <= 24 ? 24
                                       : ActiveTokens <= 32 ? 32
                                       : ActiveTokens <= 40 ? 40
                                                            : 48;
    static constexpr int kKWarps     = ActiveTokens <= 30 ? 8 : 4;
    static constexpr int kMinBlocks  = kKWarps == 8 ? 2 : 3;
    static constexpr auto kScaleAccess =
        ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    using Type = W8SmallTMmaSchedule<kKWarps, kTileTokens, kMinBlocks, kScaleAccess>;
};

template <int ActiveTokens>
struct W8LinearSmallTProductionSchedule<W835bMtpProjectionGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kW835bMtpProjectionFirstSmallT);
    static_assert(ActiveTokens <= kW835bMtpProjectionLastSmallT);

    static constexpr int kTileTokens = ActiveTokens <= 8    ? 8
                                       : ActiveTokens <= 16 ? 16
                                       : ActiveTokens <= 24 ? 24
                                       : ActiveTokens <= 32 ? 32
                                       : ActiveTokens <= 40 ? 40
                                                            : 48;
    static constexpr int kKWarps     = ActiveTokens <= 12 ? 16 : 8;
    static constexpr int kMinBlocks  = kKWarps == 16 ? 1 : 2;
    static constexpr auto kScaleAccess =
        ActiveTokens > 4 ? W8SmallTMmaScaleAccess::Shared : W8SmallTMmaScaleAccess::Direct;
    static constexpr auto kActivationCache =
        ActiveTokens == 4 || (ActiveTokens >= 27 && ActiveTokens <= 40) ? Cache::cg : Cache::ca;
    using Type =
        W8SmallTMmaSchedule<kKWarps, kTileTokens, kMinBlocks, kScaleAccess, kActivationCache>;
};

template <int ActiveTokens>
struct W8LinearSmallTProductionSchedule<W8DFlash2AttentionProjectionGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kW8DFlash2AttentionFirstSmallT);
    static_assert(ActiveTokens <= kW8DFlash2AttentionLastSmallT);

    static constexpr int kTileTokens = ActiveTokens <= 8    ? 8
                                       : ActiveTokens <= 16 ? 16
                                       : ActiveTokens <= 24 ? 24
                                       : ActiveTokens <= 32 ? 32
                                       : ActiveTokens <= 40 ? 40
                                       : ActiveTokens <= 48 ? 48
                                                            : 56;
    // RTX 5090 cold-cache winners for the complete T=1..53 interval. A longer K split wins at
    // T=1..4, eight warps cover the next two token tiles, and four warps avoid the occupancy loss
    // from T=17 onward. Direct scales remove staging overhead at T=1..4 and T=9..14.
    static constexpr int kKWarps = ActiveTokens <= 4 ? 16 : (ActiveTokens <= 16 ? 8 : 4);
    static constexpr auto kScaleAccess =
        ActiveTokens <= 4 || (ActiveTokens >= 9 && ActiveTokens <= 14)
            ? W8SmallTMmaScaleAccess::Direct
            : W8SmallTMmaScaleAccess::Shared;
    using Type = W8SmallTMmaSchedule<kKWarps, kTileTokens, 2, kScaleAccess>;
};

} // namespace ninfer::ops::detail
