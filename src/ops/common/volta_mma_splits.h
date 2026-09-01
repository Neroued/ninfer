#pragma once

// Split-K choice shared by the Volta fused-dequant tensor-core GEMMs (Q4 and Q5).
//
// Both kernels use four warps, 32 output rows per CTA, and a 32-row A tile. Their split count is
// therefore governed by one CTA-residency budget. Keeping the concurrent weight footprint below
// the L2-capacity cliff is more important than maximizing K parallelism; long-K shapes are capped
// separately for the same reason.

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

// Output rows per CTA and rows of the mma A operand, shared by Q4VoltaMmaSchedule and
// Q5VoltaMmaSchedule. Duplicated here rather than included because those live in .cuh files
// behind an sm_70 __CUDA_ARCH__ guard, and this is host-side dispatch.
inline constexpr int kVoltaMmaRowsPerCta = 32;
inline constexpr int kVoltaMmaTTile      = 32;

// Below the theoretical 640 resident CTAs so the concurrent working set remains cache-resident.
inline constexpr int kVoltaMmaCtaBudget = 384;

// Long-K shapes require a smaller concurrent footprint than the general budget implies.
inline constexpr std::int32_t kVoltaMmaLongK = 12288;

[[nodiscard]] inline int volta_mma_split_count(std::int32_t n, std::int32_t k,
                                               std::int32_t t) noexcept {
    const int row_ctas = (n + kVoltaMmaRowsPerCta - 1) / kVoltaMmaRowsPerCta;
    const int t_tiles  = (t + kVoltaMmaTTile - 1) / kVoltaMmaTTile;
    const int blocks   = row_ctas * t_tiles;
    int splits         = blocks >= kVoltaMmaCtaBudget ? 1 : kVoltaMmaCtaBudget / blocks;
    if (k >= kVoltaMmaLongK) { splits = std::min(splits, 2); }
    return std::max(splits, 1);
}

} // namespace ninfer::ops::detail
