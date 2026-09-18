#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

// 32x32 output tile with four warps. The other K=5120 shapes already instantiate this schedule;
// on this geometry it carries three to ten column tiles, where it measures faster than the wider
// schedule below.
using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

// The same tile with eight warps over a four-deep pipeline, so each warp owns a 16x8 warp tile.
// Measured against the four-warp schedule above it is faster below three column tiles and slower
// from three tiles on.
using MmaR32C32Wide = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 8, 4, 2,
                                                Q4FragmentPipeline::Serial, Cache::cg, Cache::cg,
                                                Q4ScaleLoad::Pair32>;

} // namespace

Q4Launch select_q4_n4096_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r1_q8_direct;
    // Row-split K-split capacities. Their grid is fixed at Rows/16, so they cover the extents where
    // the tiled MMA still launches too few CTAs to fill the device. Capacity 8 runs at the
    // capacity-4 latency at T=2..4, so no separate instance is kept for that point.
    if (tokens <= 8) return launch_q4_ksplit<4096, 5120, 8>;
    if (tokens <= 16) return launch_q4_ksplit<4096, 5120, 16>;
    if (tokens <= 24) return launch_q4_ksplit<4096, 5120, 24>;
    if (tokens <= 32) return launch_q4_ksplit<4096, 5120, 32>;
    // Below three column tiles the wider schedule wins; from three tiles the four-warp schedule
    // takes over and holds until the 128-column tile's CTA-wave profile wins outright.
    if (tokens <= 64) return launch_q4_mma<MmaR32C32Wide>;
    if (tokens <= 320) return launch_q4_mma<MmaR32C32>;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
