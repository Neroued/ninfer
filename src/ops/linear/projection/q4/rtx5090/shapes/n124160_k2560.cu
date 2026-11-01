#include "ops/linear/projection/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n124160_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_a16_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r32_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 64) return launch_q4_a16_mma_r64_t64_k128_s2_a1;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
