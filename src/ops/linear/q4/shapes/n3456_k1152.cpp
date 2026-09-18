#include "ops/linear/q4/q4_shapes.h"
#include <stdexcept>

namespace ninfer::ops::detail {

Q4Launch select_q4_n3456_k1152(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0) {
        throw std::invalid_argument("q4 linear: T must be a multiple of 4 in [4,131072]");
    }
    if (tokens <= 36) return launch_q4_simt_r8_c4;
    if (tokens <= 320) return launch_q4_mma_r64_c64;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
