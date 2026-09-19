#include "ops/linear/q5/q5_shapes.h"
#include "ops/linear/q5/q5_ksplit_mma.cuh"

namespace ninfer::ops::detail {

Q5Launch select_q5_n5120_k17408(std::int32_t tokens) {
    if (tokens <= 4) return launch_q5_ksplit_mma<5120, 17408, 4>;
    if (tokens <= 8) return launch_q5_ksplit_mma<5120, 17408, 8>;
    if (tokens <= 16) return launch_q5_ksplit_mma<5120, 17408, 16>;
    if (tokens <= 96) return launch_q5_ksplit_mma<5120, 17408, 32, 96>;
    if (tokens <= 112) return launch_q5_mma_r64_c32_s3;
    if (tokens <= 256) return launch_q5_mma_r32_c128;
    return launch_q5_mma_r64_c128;
}

} // namespace ninfer::ops::detail
