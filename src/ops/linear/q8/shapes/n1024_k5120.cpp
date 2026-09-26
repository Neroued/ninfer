#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

Q8Launch select_q8_n1024_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_a16_simt_r4_t4_w2_g16_s2;
    if (tokens <= 80) return launch_q8_a16_sliced_r16_t16_w8_s2;
    if (tokens <= 96) return launch_q8_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 128) return launch_q8_a16_sliced_r16_t32_w4_s2;
    return launch_q8_a16_mma_r32_t128;
}

} // namespace ninfer::ops::detail
