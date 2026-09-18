#pragma once

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

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Bf16Geometry {
    static_assert(OutputRows > 0 && InputRows > 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane,
          int AccumulatorChains, Bf16ActivationAccess ActivationAccess, Bf16WeightCache WeightCache,
          Bf16PhaseOrder PhaseOrder, int PhaseStride, int PrefetchDepth, int PhaseUnroll,
          int MinBlocksPerSm>
struct Bf16GemvSchedule {
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
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane,
          int AccumulatorChains, int TokenBatch, Bf16SimtActivationAccess ActivationAccess,
          Bf16WeightCache WeightCache, Bf16PhaseOrder PhaseOrder, int PhaseStride, int PhaseUnroll,
          int PrefetchDepth, int MinBlocksPerSm>
struct Bf16SimtSchedule {
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
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
};

} // namespace ninfer::ops::detail
