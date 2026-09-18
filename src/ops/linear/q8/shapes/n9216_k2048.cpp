#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

Q8Launch select_q8_n9216_k2048(std::int32_t tokens) {
    if (tokens <= 13) return launch_q8_simt_r8_c4;
    if (tokens <= 128) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
