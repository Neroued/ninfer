#include "ops/linear/projection/q4/q4_shapes.h"
#include "ops/linear/kernels/q4/q4_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Batch8  = Q4A16SlicedKMmaSchedule<16, 8, 8, 1, Cache::cg, Cache::ca, 2, 2560, 8>;
using Batch16 = Q4A16SlicedKMmaSchedule<16, 16, 8, 1, Cache::cg, Cache::ca, 2, 2560, 16>;
} // namespace

Q4Launch select_q4_n124160_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_q4_a16_sliced_instance<Batch8>;
    if (tokens <= 16) return launch_q4_a16_sliced_instance<Batch16>;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
