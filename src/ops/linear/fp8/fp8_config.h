#pragma once

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Fp8Geometry {
    static_assert(OutputRows > 0 && (OutputRows % 16) == 0);
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <std::int32_t InputRows>
struct Fp8ActivationGeometry {
    static_assert(InputRows > 0 && (InputRows % 32) == 0);

    static constexpr std::int32_t kInputRows = InputRows;
};

enum class Fp8CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Fp8SmallTActivationAccess : std::uint8_t {
    TokenPacked,
    SharedPhase,
};

enum class Fp8SmallTBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Fp8CodeCache CodeCache, int PhaseUnroll, int MinBlocksPerSm>
struct Fp8GemvSchedule {
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
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int TokenTile, int AccumulatorChains,
          Fp8SmallTActivationAccess ActivationAccess, Fp8CodeCache CodeCache, int PhaseUnroll,
          Fp8SmallTBlockOrder BlockOrder, int MinBlocksPerSm>
struct Fp8SmallTSchedule {
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
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
};

enum class Fp8A16SmallTMmaActivationStage : std::uint8_t {
    ActiveOnly,
    PaddedZero,
};

enum class Fp8A16SmallTMmaCache : std::uint8_t {
    Default,
    Streaming,
};

template <int KWarps, int TileTokens, int MinBlocksPerSm,
          Fp8A16SmallTMmaCache ActivationCache = Fp8A16SmallTMmaCache::Default,
          Fp8A16SmallTMmaCache WeightCache     = Fp8A16SmallTMmaCache::Streaming,
          Fp8A16SmallTMmaActivationStage ActivationStage =
              Fp8A16SmallTMmaActivationStage::ActiveOnly>
struct Fp8A16SmallTMmaSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kKWarps            = KWarps;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = KWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = KWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / KWarps;
};

using Fp8AttnInputGeometry       = Fp8Geometry<14336, 5120>;
using Fp8GdnInputGeometry        = Fp8Geometry<16384, 5120>;
using Fp8MlpGateUpGeometry       = Fp8Geometry<34816, 5120>;
using Fp8VocabularyGeometry      = Fp8Geometry<248320, 5120>;
using Fp8Residual6144Geometry    = Fp8Geometry<5120, 6144>;
using Fp8Residual17408Geometry   = Fp8Geometry<5120, 17408>;
using Fp8Activation5120Geometry  = Fp8ActivationGeometry<5120>;
using Fp8Activation6144Geometry  = Fp8ActivationGeometry<6144>;
using Fp8Activation17408Geometry = Fp8ActivationGeometry<17408>;

inline constexpr std::int32_t kFp8VocabularyFirstA16SmallTMmaT = 1;
inline constexpr std::int32_t kFp8VocabularyLastA16SmallTMmaT  = 48;
inline constexpr std::int32_t kFp8VocabularyFirstA16GemmT      = 42;

template <int ActiveTokens>
struct Fp8VocabularyA16SmallTMmaProductionSchedule {
    static_assert(ActiveTokens >= kFp8VocabularyFirstA16SmallTMmaT);
    static_assert(ActiveTokens <= kFp8VocabularyLastA16SmallTMmaT);

    static constexpr int kTileTokens     = ActiveTokens <= 8    ? 8
                                           : ActiveTokens <= 16 ? 16
                                           : ActiveTokens <= 24 ? 24
                                           : ActiveTokens <= 32 ? 32
                                           : ActiveTokens <= 40 ? 40
                                                                : 48;
    static constexpr int kKWarps         = ActiveTokens <= 8 ? 16 : (ActiveTokens <= 24 ? 8 : 4);
    static constexpr int kMinBlocksPerSm = kKWarps == 16 ? 1 : 2;
    using Type = Fp8A16SmallTMmaSchedule<kKWarps, kTileTokens, kMinBlocksPerSm>;
};

enum class Fp8Problem : std::uint8_t {
    AttnInput,
    GdnInput,
    MlpGateUp,
    Vocabulary,
    Residual6144,
    Residual17408,
};

inline constexpr bool is_fp8_linear_problem(std::int32_t output_rows, std::int32_t input_rows) {
    return (output_rows == Fp8AttnInputGeometry::kOutputRows &&
            input_rows == Fp8AttnInputGeometry::kInputRows) ||
           (output_rows == Fp8GdnInputGeometry::kOutputRows &&
            input_rows == Fp8GdnInputGeometry::kInputRows) ||
           (output_rows == Fp8MlpGateUpGeometry::kOutputRows &&
            input_rows == Fp8MlpGateUpGeometry::kInputRows) ||
           (output_rows == Fp8VocabularyGeometry::kOutputRows &&
            input_rows == Fp8VocabularyGeometry::kInputRows) ||
           (output_rows == Fp8Residual6144Geometry::kOutputRows &&
            input_rows == Fp8Residual6144Geometry::kInputRows) ||
           (output_rows == Fp8Residual17408Geometry::kOutputRows &&
            input_rows == Fp8Residual17408Geometry::kInputRows);
}

inline Fp8Problem resolve_fp8_problem(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Fp8AttnInputGeometry::kOutputRows &&
        input_rows == Fp8AttnInputGeometry::kInputRows) {
        return Fp8Problem::AttnInput;
    }
    if (output_rows == Fp8GdnInputGeometry::kOutputRows &&
        input_rows == Fp8GdnInputGeometry::kInputRows) {
        return Fp8Problem::GdnInput;
    }
    if (output_rows == Fp8MlpGateUpGeometry::kOutputRows &&
        input_rows == Fp8MlpGateUpGeometry::kInputRows) {
        return Fp8Problem::MlpGateUp;
    }
    if (output_rows == Fp8VocabularyGeometry::kOutputRows &&
        input_rows == Fp8VocabularyGeometry::kInputRows) {
        return Fp8Problem::Vocabulary;
    }
    if (output_rows == Fp8Residual6144Geometry::kOutputRows &&
        input_rows == Fp8Residual6144Geometry::kInputRows) {
        return Fp8Problem::Residual6144;
    }
    if (output_rows == Fp8Residual17408Geometry::kOutputRows &&
        input_rows == Fp8Residual17408Geometry::kInputRows) {
        return Fp8Problem::Residual17408;
    }
    throw std::invalid_argument("unsupported FP8 problem");
}

template <class Geometry>
struct Fp8LinearDecodeProductionSchedule;

// RTX 5090 cold-cache winner for this exact problem. Each newly registered geometry supplies its
// own specialization so admission never silently inherits another problem's measured schedule.
template <>
struct Fp8LinearDecodeProductionSchedule<Fp8AttnInputGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8GdnInputGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8MlpGateUpGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

#ifdef NINFER_VOLTA_BUILD
template <>
struct Fp8LinearDecodeProductionSchedule<Fp8VocabularyGeometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};
#endif

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8Residual6144Geometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

template <>
struct Fp8LinearDecodeProductionSchedule<Fp8Residual17408Geometry> {
    using Type = Fp8GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
};

inline constexpr std::int32_t kFp8FirstSmallT = 2;
inline constexpr std::int32_t kFp8LastSmallT  = 24;

template <class Geometry>
inline constexpr std::int32_t kFp8LinearSmallTMax = 0;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8AttnInputGeometry> = 11;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8GdnInputGeometry> = 10;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8MlpGateUpGeometry> = 4;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8Residual6144Geometry> = kFp8LastSmallT;

template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8Residual17408Geometry> = kFp8LastSmallT;

#ifdef NINFER_VOLTA_BUILD
template <>
inline constexpr std::int32_t kFp8LinearSmallTMax<Fp8VocabularyGeometry> = kFp8LastSmallT;
#endif

inline std::int32_t fp8_linear_small_t_max(Fp8Problem problem) {
    switch (problem) {
    case Fp8Problem::AttnInput:
        return kFp8LinearSmallTMax<Fp8AttnInputGeometry>;
    case Fp8Problem::GdnInput:
        return kFp8LinearSmallTMax<Fp8GdnInputGeometry>;
    case Fp8Problem::MlpGateUp:
        return kFp8LinearSmallTMax<Fp8MlpGateUpGeometry>;
    case Fp8Problem::Vocabulary:
#ifdef NINFER_VOLTA_BUILD
        return kFp8LinearSmallTMax<Fp8VocabularyGeometry>;
#else
        break;
#endif
    case Fp8Problem::Residual6144:
        return kFp8LinearSmallTMax<Fp8Residual6144Geometry>;
    case Fp8Problem::Residual17408:
        return kFp8LinearSmallTMax<Fp8Residual17408Geometry>;
    }
    throw std::logic_error("FP8 vocabulary uses its A16 MMA route");
}

// RTX 5090 cold-cache winners for contiguous Linear output. Each geometry owns its measured
// schedule ranges; fused semantic Ops reuse the mainloop but retain independent route frontiers.
template <class Geometry, int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Geometry>);
    static constexpr int kValuesPerLane     = 16;
    static constexpr auto kActivationAccess = Fp8SmallTActivationAccess::TokenPacked;
    using Type =
        Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                          Fp8CodeCache::Default, 1, Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8AttnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8AttnInputGeometry>);
    static constexpr auto kActivationAccess = ActiveTokens >= 3 && ActiveTokens <= 4
                                                  ? Fp8SmallTActivationAccess::SharedPhase
                                                  : Fp8SmallTActivationAccess::TokenPacked;
    using Type =
        Fp8SmallTSchedule<8, 2, 16, ActiveTokens, 1, kActivationAccess, Fp8CodeCache::Default, 1,
                          Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8GdnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8GdnInputGeometry>);
    static constexpr int kValuesPerLane     = ActiveTokens >= 5 && ActiveTokens <= 6 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Fp8SmallTActivationAccess::SharedPhase
                                                  : Fp8SmallTActivationAccess::TokenPacked;
    using Type =
        Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                          Fp8CodeCache::Default, 1, Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8MlpGateUpGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8MlpGateUpGeometry>);
    static constexpr int kValuesPerLane     = ActiveTokens == 4 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 3
                                                  ? Fp8SmallTActivationAccess::SharedPhase
                                                  : Fp8SmallTActivationAccess::TokenPacked;
    using Type =
        Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                          Fp8CodeCache::Default, 1, Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8Residual6144Geometry>);
    static constexpr int kValuesPerLane = ActiveTokens >= 20 && ActiveTokens <= 23 ? 8 : 16;
    static constexpr int kTokenTile     = ActiveTokens == 24 ? 12 : ActiveTokens;
    static constexpr auto kBlockOrder   = ActiveTokens == 24
                                              ? Fp8SmallTBlockOrder::TokenTilesContiguous
                                              : Fp8SmallTBlockOrder::RowsContiguous;

    using Type = Fp8SmallTSchedule<8, 2, kValuesPerLane, kTokenTile, 1,
                                   Fp8SmallTActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                   kBlockOrder, 1>;
};

template <int ActiveTokens>
struct Fp8LinearSmallTProductionSchedule<Fp8Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kFp8FirstSmallT);
    static_assert(ActiveTokens <= kFp8LinearSmallTMax<Fp8Residual17408Geometry>);
    static constexpr int kValuesPerLane = ActiveTokens >= 18 ? 8 : 16;

    using Type = Fp8SmallTSchedule<8, 2, kValuesPerLane, ActiveTokens, 1,
                                   Fp8SmallTActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                   Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

} // namespace ninfer::ops::detail
