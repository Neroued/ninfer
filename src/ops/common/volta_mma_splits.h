#pragma once

// Split-K choice shared by the Volta fused-dequant tensor-core GEMMs (Q4 and Q5).
//
// Both kernels use the same schedule -- 4 warps, 32 output rows per CTA, a 32-row A tile -- so
// they make the same tradeoff, and it is not the one the first tuning pass assumed. The earlier
// tables keyed off K (Q5) or were a single constant (Q4, fixed at 4 from one N=4096 sweep), and
// both extrapolate badly: measured on a V100-SXM2-32GB at T=24 with the full launcher (memset +
// main kernel + narrowing), us --
//
//   Q4 k=5120   splits:      1        2        4        8
//     n= 4096              191.5    129.0     96.3    101.4      -> 4
//     n= 6144              ...      151.6    ...      ...        -> 2
//     n= 8192              ...      165.9    ...      ...        -> 2
//     n=12288              246.8    ...      ...      ...        -> 1
//     n=34816              702.5    722.0    753.7    878.6      -> 1   (table said 4: -7.3%)
//   Q5
//     n=5120 k= 6144       265.2    162.9    194.6    178.2      -> 3 (149.5)  (table said 8)
//     n=6144 k= 5120       216.1    153.6    214.0    179.2      -> 2         (table said 8)
//     n=5120 k=17408       700.4    435.1   1041.4    803.8      -> 2         (table said 32)
//
// The controlling variable is neither K nor CTA count for its own sake, but L2 hit rate. ncu on
// the Q5 n=5120 k=17408 case, one launch each:
//
//   splits:                    1       2       4       8      16      32
//   dram__bytes_read      60.0MB  69.5MB  396.7MB 280.9MB 193.0MB 102.1MB
//   lts__t_sector_hit_rate  86.0%   90.6%    49.7%   64.4%   77.0%   88.5%
//   duration                763us   478us   1060us   812us   659us   549us
//
// The weight is 58.5 MB. At the right split count it is read essentially once; at splits=4 it is
// read about seven times. Duration tracks DRAM bytes, not occupancy -- splits=4 has *twice* the
// achieved occupancy of splits=2 (49.5% vs 24.8% warps active) and is 2.2x slower. So more K
// parallelism is not free once the launch stops fitting in one resident wave: the extra CTAs sit
// at different K offsets, the concurrent read footprint stops fitting in the 6 MB L2, and the
// kernel goes back to DRAM for weight bytes it used to hit on.
//
// Hence the budget below. The V100 holds 80 SMs x 8 CTAs = 640 of these resident (ncu confirms
// the register limit is 8 blocks at 64 registers/thread); the measured optimum sits at 384-512,
// just under that, for every shape swept. `kCtaBudget` is the largest launch that stayed on the
// good side of the cliff.
//
// Known deviation, deliberately not encoded: at T=48/64 (t_tiles=2) the n=5120 k=6144 shape wants
// 2 where the budget says 1 (249.9us vs 285.7us). That is C16 territory; the tuned configurations
// run T=12-32, where the budget is exact for every production shape.

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {

// Output rows per CTA and rows of the mma A operand, shared by Q4VoltaMmaSchedule and
// Q5VoltaMmaSchedule. Duplicated here rather than included because those live in .cuh files
// behind an sm_70 __CUDA_ARCH__ guard, and this is host-side dispatch.
inline constexpr int kVoltaMmaRowsPerCta = 32;
inline constexpr int kVoltaMmaTTile      = 32;

// Below 640 resident CTAs on purpose; see the L2 evidence above.
//
// Retuned from 512 to 384 when the kernels were software-pipelined. That change makes each CTA
// better at hiding its own global latency, which shifts the balance towards fewer and longer
// CTAs: three of the five production shapes moved down one step (Q4 n=4096 4->3, Q5 n=5120
// k=6144 3->2, Q5 n=1024 16->8) for 2.3-3.6% each, and 384 is the budget that reproduces every
// new optimum.
inline constexpr int kVoltaMmaCtaBudget = 384;

// Long-K shapes want a smaller concurrent footprint than the budget alone implies: at
// n=5120 k=17408 T=24 the budget picks 3 (624.6us) where 2 measures 435.1us.
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
