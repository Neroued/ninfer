#include "ops/linear/projection/q8/rtx5090/shapes.h"

namespace ninfer::ops::detail {

Q8Launch select_q8_n12288_k2048(std::int32_t tokens) {
    if (tokens <= 16) return launch_q8_a16_simt_r8_t4;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
