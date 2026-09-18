#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

// Field-for-field the instantiations other Q4 shapes already compile (n1024_k5120.cu,
// n4096_k5120.cu, n6144_k5120.cu, n7168_k5120.cu), so these add no compiled instance.
using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

Q4Launch select_q4_n34816_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r1_q8_direct;
    // The K-split tile is masked to the capacity rounded up to a multiple of eight, so each
    // capacity below is the narrowest masked tile that still covers its interval.
    if (tokens <= 4) return launch_q4_ksplit<34816, 5120, 4>;
    if (tokens <= 8) return launch_q4_ksplit<34816, 5120, 8>;
    if (tokens <= 16) return launch_q4_ksplit<34816, 5120, 16>;
    if (tokens <= 24) return launch_q4_ksplit<34816, 5120, 24>;
    // Above the K-split ceiling the narrowest column tile that covers the extent wins: at these
    // row counts the 128-wide tile that served every T >= 17 before this change spends a whole
    // 128-column tile on T = 17. Each bound is a tile width, not a threshold copied from another
    // shape; the measured cost of merging an interval into its neighbour is in the PR body.
    if (tokens <= 32) return launch_q4_mma<MmaR32C32>;
    if (tokens <= 48) return launch_q4_mma_r64_c48;
    if (tokens <= 56) return launch_q4_mma_r64_c56;
    if (tokens <= 64) return launch_q4_mma<MmaR32C64>;
    if (tokens <= 80) return launch_q4_mma_r64_c80;
    if (tokens <= 96) return launch_q4_mma_r64_c96;
    if (tokens <= 120) return launch_q4_mma_r64_c120;
    if (tokens <= 128) return launch_q4_mma_r64_c128;
    // Above the hot interval a route is applied once per column tile, so the winner is the tile
    // that splits the extent into the fewest slices, and among those the narrowest: two tiles of
    // 96 cover 129..192, of 112 cover 193..224, of 120 cover 225..240 and of 128 cover 241..256;
    // three tiles of 96 then beat three wider ones up to 288, and 128 carries the 512 and 1024
    // anchors. Each bound is a tile-count edge, not a threshold copied from another shape; the
    // measured cost of folding one of these bands into its neighbour is in the PR body.
    if (tokens <= 192) return launch_q4_mma_r64_c96;
    if (tokens <= 224) return launch_q4_mma_r64_c112;
    if (tokens <= 240) return launch_q4_mma_r64_c120;
    if (tokens <= 256) return launch_q4_mma_r64_c128;
    if (tokens <= 288) return launch_q4_mma_r64_c96;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
