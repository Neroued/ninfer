#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n34816_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q6_simt_r8_c4;
    if (tokens <= 5) return launch_q6_simt_r8_c5;
    if (tokens <= 6) return launch_q6_simt_r8_c6;
    if (tokens <= 7) return launch_q6_simt_r8_c7;
    if (tokens <= 16) return launch_q6_mma_r64_c16_k128;
    if (tokens <= 24) return launch_q6_mma_r64_c24_k128;
    if (tokens <= 32) return launch_q6_mma_r64_c32_k128;
    if (tokens <= 48) return launch_q6_mma_r64_c48_k128;
    return launch_q6_mma_r64_c128;
}

} // namespace ninfer::ops::detail
