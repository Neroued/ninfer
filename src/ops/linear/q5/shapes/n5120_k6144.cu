#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"

namespace ninfer::ops::detail {

Q5Launch select_q5_n5120_k6144(std::int32_t tokens) {
    if (tokens == 1) return launch_q5_split4_c1_k6144;
    if (tokens <= 96) return launch_q5_ksplit_mma<5120, 6144, 32, 96>;
    if (tokens <= 112) return launch_q5_mma_r64_c32_s3;
    if (tokens <= 256) return launch_q5_mma_r32_c128;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
