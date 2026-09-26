#include "ops/linear/q6/q6_shapes.h"
#include <stdexcept>

namespace ninfer::ops::detail {

Q6Launch select_q6_n1152_k1536(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0) {
        throw std::invalid_argument("q6 linear: T must be a multiple of 4 in [4,131072]");
    }
    if (tokens <= 4) return launch_q6_a16_simt_r8_t4;
    if (tokens <= 16) return launch_q6_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 24) return launch_q6_a16_sliced_r16_t24_w4_s2;
    if (tokens <= 80) return launch_q6_a16_sliced_r16_t32_w4_s2;
    if (tokens <= 384) return launch_q6_a16_sliced_r32_t32_w4_s2;
    // This wider capacity wins both bulk anchors; the tiled MMA wins beyond this wave interval.
    if (tokens <= 1152) return launch_q6_a16_sliced_r32_t64_w2_s1;
    return launch_q6_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
