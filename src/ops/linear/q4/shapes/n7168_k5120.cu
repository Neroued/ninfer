#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

// Field-for-field the instantiations other Q4 shapes already compile (n4096_k5120.cu,
// n6144_k5120.cu), so these add no compiled instance.
using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C32Wide = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 8, 4, 2, Q4FragmentPipeline::Serial,
                                                Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C64 = Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 16, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

Q4Launch select_q4_n7168_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r1_q8_direct;
    if (tokens <= 8) return launch_q4_ksplit<7168, 5120, 8>;
    if (tokens <= 16) return launch_q4_ksplit<7168, 5120, 16>;
    if (tokens <= 24) return launch_q4_ksplit<7168, 5120, 24>;
    if (tokens <= 64) return launch_q4_mma<MmaR32C32Wide>;
    if (tokens <= 96) return launch_q4_mma<MmaR32C32>;
    if (tokens <= 112) return launch_q4_mma<MmaR32C64>;
    if (tokens <= 128) return launch_q4_mma_r64_c48;
    // Measured windows above the hot interval: the 512 anchor and the tile-count bands around it
    // prefer r64_c96, and the 129..192 and 385..448 bands prefer the 64x64 tile.
    if (tokens <= 192) return launch_q4_mma<MmaR64C64>;
    if (tokens <= 288) return launch_q4_mma_r64_c96;
    if (tokens <= 384) return launch_q4_mma_r64_c128;
    if (tokens <= 448) return launch_q4_mma<MmaR64C64>;
    if (tokens <= 576) return launch_q4_mma_r64_c96;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
