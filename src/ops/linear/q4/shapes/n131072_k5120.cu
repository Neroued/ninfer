#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {

Q4Launch select_q4_n131072_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r4_w1_direct;
    if (tokens <= 4) return launch_q4_ksplit<131072, 5120, 4>;
    if (tokens <= 8) return launch_q4_ksplit<131072, 5120, 8>;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
