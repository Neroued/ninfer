#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR64C64 = Q4RowSplitMmaGemmSchedule<64, 64, 64, 32, 16, 2, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

Q4Launch select_q4_n6144_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r1_q8_direct;
    if (tokens <= 8) return launch_q4_ksplit<6144, 5120, 8>;
    if (tokens <= 16) return launch_q4_ksplit<6144, 5120, 16>;
    if (tokens <= 24) return launch_q4_ksplit<6144, 5120, 24>;
    if (tokens <= 96) return launch_q4_mma<MmaR32C32>;
    if (tokens <= 192) return launch_q4_mma<MmaR32C64>;
    // This tile improves the 512 anchor and its surrounding CTA-wave interval.
    if (tokens > 384 && tokens <= 640) return launch_q4_mma<MmaR64C64>;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
