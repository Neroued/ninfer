#include "ops/linear/q8/q8_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q8Launch select_q8_n4608_k4608(std::int32_t tokens) {
    if (tokens > 32768) throw std::invalid_argument("q8 linear: column extent exceeds 32768");
    if (tokens <= 8 || (tokens <= 12 && tokens % 4 == 0)) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 256) return launch_q8_a16_mma_r32_t128;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
