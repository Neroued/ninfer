#include "ops/linear/q6/q6_shapes.h"
#include <stdexcept>

namespace ninfer::ops::detail {

Q6Launch select_q6_n1152_k1536(std::int32_t tokens) {
    if (tokens > 131072 || tokens % 4 != 0) {
        throw std::invalid_argument("q6 linear: T must be a multiple of 4 in [4,131072]");
    }
    if (tokens <= 96) return launch_q6_simt_r8_c4;
    if (tokens <= 704) return launch_q6_mma_r64_c64;
    return launch_q6_mma_r64_c128;
}

} // namespace ninfer::ops::detail
