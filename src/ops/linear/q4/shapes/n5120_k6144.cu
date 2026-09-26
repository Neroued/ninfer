#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {
Q4Launch select_q4_n5120_k6144(std::int32_t tokens) {
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 1) return launch_q4_a16_gemv_r1_w8_k6144;
    if (tokens <= 4) return launch_q4_a16_sliced_r16_t8_capacity4;
    if (tokens <= 8) return launch_q4_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q4_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 96) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 128) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 192) return launch_q4_a16_mma_r32_t64_k64_wr16_wt32_s3_a3_b2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
