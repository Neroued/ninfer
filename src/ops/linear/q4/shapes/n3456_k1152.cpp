#include "ops/linear/q4/q4_shapes.h"
#include <stdexcept>

namespace ninfer::ops::detail {
Q4Launch select_q4_n3456_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0)
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    // Selected with complete-Op cold CUDA Graph measurements on RTX 5090.
    if (tokens <= 16) return launch_q4_a16_sliced_r32_t16_w4_s2;
    if (tokens <= 64) return launch_q4_a16_sliced_r16_t16_w2_s2;
    if (tokens <= 96) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 128) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 192) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 256) return launch_q4_a16_mma_r32_t32_k128_s2_a2;
    if (tokens <= 320) return launch_q4_a16_sliced_r32_t32_w2_s2;
    if (tokens <= 384) return launch_q4_a16_mma_r32_t128_k64_s2_a2;
    return launch_q4_a16_mma_r64_t128;
}
} // namespace ninfer::ops::detail
