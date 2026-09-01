#pragma once

#include "ninfer/ops/linear.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

// Several NVFP4 plans drive a sub-projection internally and choose the policy themselves rather
// than taking the caller's, so a target that asks for A16 cannot reach them. On Volta the A4
// route is a __trap() -- cvt.e2m1x2 is Blackwell hardware and always will be -- so those internal
// choices have to collapse to A16 here. See docs/v100.md.
#ifdef NINFER_VOLTA_BUILD
inline constexpr LinearPolicy kNvfp4InternalPolicy = LinearPolicy::A16Only;
#else
inline constexpr LinearPolicy kNvfp4InternalPolicy = LinearPolicy::AllowA4;
#endif

enum class Nvfp4ScaleAccess : std::uint8_t {
    StagedRaw,
    Direct,
};

enum class Nvfp4CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Nvfp4SmallTActivationAccess : std::uint8_t {
    PairStream,
    TokenPacked,
    SharedPhase,
};

enum class Nvfp4SmallTBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4GemvGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int MinBlocksPerSm>
struct Nvfp4GemvSchedule {
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
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane, int TokenTile,
          int AccumulatorChains, Nvfp4SmallTActivationAccess ActivationAccess,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int PhaseUnroll,
          Nvfp4SmallTBlockOrder BlockOrder, int MinBlocksPerSm>
struct Nvfp4SmallTSchedule {
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
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

using Nvfp4AttnInputGeometry     = Nvfp4GemvGeometry<14336, 5120>;
using Nvfp4GdnInputGeometry      = Nvfp4GemvGeometry<16384, 5120>;
using Nvfp4MlpGateUpGeometry     = Nvfp4GemvGeometry<34816, 5120>;
using Nvfp4Residual6144Geometry  = Nvfp4GemvGeometry<5120, 6144>;
using Nvfp4Residual17408Geometry = Nvfp4GemvGeometry<5120, 17408>;

using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;

enum class Nvfp4Problem : std::uint8_t {
    AttnInput,
    GdnInput,
    MlpGateUp,
    Residual6144,
    Residual17408,
};

inline constexpr bool is_nvfp4_linear_problem(std::int32_t output_rows, std::int32_t input_rows) {
    return (output_rows == Nvfp4AttnInputGeometry::kOutputRows &&
            input_rows == Nvfp4AttnInputGeometry::kInputRows) ||
           (output_rows == Nvfp4GdnInputGeometry::kOutputRows &&
            input_rows == Nvfp4GdnInputGeometry::kInputRows) ||
           (output_rows == Nvfp4MlpGateUpGeometry::kOutputRows &&
            input_rows == Nvfp4MlpGateUpGeometry::kInputRows) ||
           (output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
            input_rows == Nvfp4Residual6144Geometry::kInputRows) ||
           (output_rows == Nvfp4Residual17408Geometry::kOutputRows &&
            input_rows == Nvfp4Residual17408Geometry::kInputRows);
}

inline Nvfp4Problem resolve_nvfp4_problem(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Nvfp4AttnInputGeometry::kOutputRows &&
        input_rows == Nvfp4AttnInputGeometry::kInputRows) {
        return Nvfp4Problem::AttnInput;
    }
    if (output_rows == Nvfp4GdnInputGeometry::kOutputRows &&
        input_rows == Nvfp4GdnInputGeometry::kInputRows) {
        return Nvfp4Problem::GdnInput;
    }
    if (output_rows == Nvfp4MlpGateUpGeometry::kOutputRows &&
        input_rows == Nvfp4MlpGateUpGeometry::kInputRows) {
        return Nvfp4Problem::MlpGateUp;
    }
    if (output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
        input_rows == Nvfp4Residual6144Geometry::kInputRows) {
        return Nvfp4Problem::Residual6144;
    }
    if (output_rows == Nvfp4Residual17408Geometry::kOutputRows &&
        input_rows == Nvfp4Residual17408Geometry::kInputRows) {
        return Nvfp4Problem::Residual17408;
    }
    throw std::invalid_argument("unsupported NVFP4 problem");
}

// RTX 5090 cold-cache winner among the measured decode schedules.
template <class Geometry>
struct Nvfp4LinearDecodeProductionSchedule {
    using Type =
        Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
};

inline constexpr std::int32_t kNvfp4FirstSmallT = 2;
inline constexpr std::int32_t kNvfp4LastSmallT  = 32;

#ifdef NINFER_VOLTA_BUILD
// Qualified Volta schedule shared by all five registered geometries. Their common performance
// envelope does not justify per-geometry specializations.
//
// The whole envelope is per-thread register footprint. Values-per-lane 8 rather than 16 is
// worth 2.1x at T=13 and 4.3x at T=32; with 16 values a 16-warp CTA cannot be resident at all
// past T=13, while with 8 it is the fastest choice at T=14..20. Accumulator chains were swept
// and never won: unlike the tensor-core routes this kernel is not dependency-bound.
template <class Geometry, int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 3    ? 16
                                          : ActiveTokens <= 13 ? 4
                                          : ActiveTokens <= 20 ? 16
                                                               : 4;
    static constexpr int kValuesPerLane = ActiveTokens <= 3 ? 16 : 8;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

#else // NINFER_VOLTA_BUILD

// RTX 5090 cold-cache winners for contiguous Linear output. T=2..4 amortizes activation loads
// through shared staging; T=5..32 keeps one packed activation tile per warp. The warp-count changes
// are measured occupancy/register crossovers, not semantic frontiers.
template <class Geometry, int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 13 ? 16 : 8);
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// G1's wider N benefits from keeping four warps per CTA throughout the A16 policy boundary. Only
// T=2 amortizes activation traffic enough for shared staging to win.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens == 2
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// At N=5120, R1 needs the larger CTA only for the last three A16 token counts. The unoptimized
// A16-only tail keeps the established generic schedule.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 16 ? (ActiveTokens >= 14 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// R2's longer K moves the stable four-to-sixteen-warp crossover to T=8.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens <= 16 ? (ActiveTokens >= 8 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

#endif // NINFER_VOLTA_BUILD

} // namespace ninfer::ops::detail
