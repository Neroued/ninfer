#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"

namespace ninfer::ops::detail {

Q5Launch select_q5_n7168_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_split4_c1_k5120;
    if (tokens <= 96) return launch_q5_ksplit_mma<7168, 5120, 32, 96>;
    if (tokens <= 112) return launch_q5_mma_r64_c32_s3;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
