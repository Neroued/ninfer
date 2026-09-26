#include "ops/linear/q5/q5_shapes.h"
#include <stdexcept>

namespace ninfer::ops::detail {
Q5Launch select_q5_n1152_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q5 linear: T must be a multiple of 4 in [4,131072]");
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 8) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 64) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 128) return launch_q5_a16_sliced_r16_t16_w2_s2;
    if (tokens <= 576) return launch_q5_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 1152) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
