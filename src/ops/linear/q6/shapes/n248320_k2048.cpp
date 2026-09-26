#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n248320_k2048(std::int32_t tokens) {
    // Capacity routes come from complete-Op cold Graph comparisons on RTX 5090.
    if (tokens <= 1) return launch_q6_a16_gemv_r4_w2_g16;
    if (tokens <= 16) return launch_q6_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 32) return launch_q6_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 40) return launch_q6_a16_mma_r64_t40_k128;
    if (tokens <= 48) return launch_q6_a16_mma_r64_t48_k128;
    if (tokens <= 56) return launch_q6_a16_mma_r64_t56_k128;
    if (tokens <= 64) return launch_q6_a16_sliced_r32_t64_w2_s1;
    if (tokens <= 72) return launch_q6_a16_mma_r64_t72_k128;
    if (tokens <= 80) return launch_q6_a16_mma_r64_t80;
    if (tokens <= 96) return launch_q6_a16_mma_r64_t96;
    if (tokens <= 112) return launch_q6_a16_mma_r64_t112;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
