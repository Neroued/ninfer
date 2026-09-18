#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {

Q4Launch select_q4_n131072_k2048(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_ksplit<131072, 2048, 4>;
    if (tokens <= 8) return launch_q4_ksplit<131072, 2048, 8>;
    if (tokens <= 16) return launch_q4_ksplit<131072, 2048, 16>;
    if (tokens <= 20) return launch_q4_ksplit<131072, 2048, 20>;
    if (tokens <= 32) return launch_q4_mma_r64_c32;
    if (tokens <= 48) return launch_q4_mma_r64_c48;
    if (tokens <= 56) return launch_q4_mma_r64_c56;
    if (tokens <= 72) return launch_q4_mma_r64_c72;
    if (tokens <= 80) return launch_q4_mma_r64_c80;
    if (tokens <= 96) return launch_q4_mma_r64_c96;
    if (tokens <= 112) return launch_q4_mma_r64_c112;
    if (tokens <= 120) return launch_q4_mma_r64_c120;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
