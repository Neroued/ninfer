#include "ops/linear/q5/q5_shapes.h"
#include <stdexcept>

namespace ninfer::ops::detail {
Q5Launch select_q5_n1152_k4304(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q5 linear: T must be a multiple of 4 in [4,131072]");
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 4) return launch_q5_a16_direct_r2_t4_w4_g4_b4;
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 32) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 48) return launch_q5_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 96) return launch_q5_a16_sliced_r32_t24_w4_s2_pairwise;
    if (tokens <= 288) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 448) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 576) return launch_q5_a16_sliced_r32_t64_w2_s2;
    if (tokens <= 672) return launch_q5_a16_sliced_r32_t32_w4_s1;
    if (tokens <= 1152) return launch_q5_a16_mma_r32_t128;
    if (tokens <= 1536) return launch_q5_a16_mma_r64_t96_k128_s1_a1;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
