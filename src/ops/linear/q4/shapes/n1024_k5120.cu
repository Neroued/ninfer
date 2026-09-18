#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_mma_launch.cuh"
#include "ops/linear/q4/q4_simt_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using SimtR4C4 = Q4RowSplitSimtGemmSchedule<4, 4, 8, 2, Cache::ca, 1>;
static_assert((5120 / 64) % SimtR4C4::kGroupsPerStage == 0);
using MmaR16C32 = Q4RowSplitMmaGemmSchedule<16, 32, 64, 16, 8, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

Q4Launch select_q4_n1024_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r1_q8_direct;
    if (tokens <= 56) return launch_q4_simt<SimtR4C4, true>;
    if (tokens <= 320) return launch_q4_mma<MmaR16C32>;
    if (tokens <= 1344) return launch_q4_mma<MmaR32C64>;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
