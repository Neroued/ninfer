#include "ops/linear/q6/q6_shapes.h"

namespace ninfer::ops::detail {

Q6Launch select_q6_n248320_k2048(std::int32_t tokens) {
    if (tokens <= 3) return launch_q6_simt_r8_c4;
    if (tokens <= 16) return launch_q6_mma_r64_c16_k128;
    if (tokens <= 24) return launch_q6_mma_r64_c24_k128;
    if (tokens <= 32) return launch_q6_mma_r64_c32_k128;
    if (tokens <= 40) return launch_q6_mma_r64_c40_k128;
    if (tokens <= 48) return launch_q6_mma_r64_c48_k128;
    if (tokens <= 56) return launch_q6_mma_r64_c56_k128;
    if (tokens <= 64) return launch_q6_mma_r64_c64_k128;
    if (tokens <= 72) return launch_q6_mma_r64_c72_k128;
    if (tokens <= 80) return launch_q6_mma_r64_c80;
    if (tokens <= 96) return launch_q6_mma_r64_c96;
    if (tokens <= 112) return launch_q6_mma_r64_c112;
    return launch_q6_mma_r64_c128;
}

} // namespace ninfer::ops::detail
