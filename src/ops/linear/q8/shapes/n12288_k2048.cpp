#include "ops/linear/q8/q8_shapes.h"

namespace ninfer::ops::detail {

Q8Launch select_q8_n12288_k2048(std::int32_t tokens) {
    if (tokens <= 16) return launch_q8_simt_r8_c4;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
