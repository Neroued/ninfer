#include "ops/linear/projection/q4/q4_shapes.h"
#include "ops/linear/kernels/q4/q4_instance_launch.cuh"

namespace ninfer::ops::detail {
namespace {
template <class Schedule, int Rows = 248320, int K = 5120>
struct HeadSchedule : Schedule {
    static constexpr bool kIndependentTokenTail = true;
};
} // namespace

Q4Launch select_q4_n248320_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s1;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 80)
        return launch_q4_a16_mma_instance<HeadSchedule<q4_instances::MmaR64T80K64Wr16Wt40S2A2B1>>;
    if (tokens <= 96)
        return launch_q4_a16_mma_instance<HeadSchedule<q4_instances::MmaR64T96K64Wr32Wt16S2A2B1>>;
    return launch_q4_a16_mma_instance<HeadSchedule<q4_instances::MmaR64T128K64Wr64Wt32S2A2B1>>;
}
} // namespace ninfer::ops::detail
