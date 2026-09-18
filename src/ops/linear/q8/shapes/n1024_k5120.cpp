#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

Q8Launch select_q8_n1024_k5120(std::int32_t tokens) {
    if (tokens <= 4) return launch_q8_simt_r8_c4;
    if (tokens <= 16) return launch_q8_simt_r8_c8;
    return launch_q8_mma_r32_c128;
}

} // namespace ninfer::ops::detail
