#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_gemv_launch.cuh"
#include "ops/linear/q4/q4_ksplit_launch.cuh"
#include "ops/linear/q4/q4_mma_launch.cuh"

namespace ninfer::ops::detail {
namespace {

using GemvR1W8 =
    Q4RowSplitGemvSchedule<1, 8, 16, 1, Q4GemvActivationAccess::Direct,
                           Q4GemvLaneMapping::PackedByte2, Q4GemvDecodeMode::ScalarInteger,
                           Q4GemvCodeTransfer::SyncVector16, Q4GemvScaleAccess::Scalar16Shuffle,
                           Cache::ca, 6144 / 64, 1>;
using MmaR32C32 = Q4RowSplitMmaGemmSchedule<32, 32, 64, 16, 16, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;
using MmaR32C64 = Q4RowSplitMmaGemmSchedule<32, 64, 64, 16, 32, 3, 2, Q4FragmentPipeline::Serial,
                                            Cache::cg, Cache::cg, Q4ScaleLoad::Pair32>;

} // namespace

Q4Launch select_q4_n5120_k6144(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv<GemvR1W8>;
    if (tokens <= 4) return launch_q4_ksplit<5120, 6144, 4>;
    if (tokens <= 8) return launch_q4_ksplit<5120, 6144, 8>;
    if (tokens <= 16) return launch_q4_ksplit<5120, 6144, 16>;
    if (tokens <= 24) return launch_q4_ksplit<5120, 6144, 24>;
    if (tokens <= 32) return launch_q4_ksplit<5120, 6144, 32>;
    if (tokens <= 96) return launch_q4_mma<MmaR32C32>;
    if (tokens <= 192) return launch_q4_mma<MmaR32C64>;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
