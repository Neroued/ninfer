#include "ops/linear/q8/q8_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q8Launch select_q8_n2048_k4608(std::int32_t tokens) {
    if (tokens > 32768) throw std::invalid_argument("q8 linear: column extent exceeds 32768");
    if (tokens <= 14 || (tokens <= 32 && tokens % 4 == 0)) return launch_q8_simt_r8_c4;
    if (tokens <= 871) return launch_q8_mma_r32_c128;
    return launch_q8_mma_r64_c128;
}

} // namespace ninfer::ops::detail
