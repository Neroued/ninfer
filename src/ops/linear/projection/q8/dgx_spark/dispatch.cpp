#include "ops/linear/projection/q8/q8_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q8Launch select_q8_a16_launch(std::int32_t, std::int32_t, std::int32_t) {
    throw std::invalid_argument("q8 linear: no implementation for dgx_spark");
}

} // namespace ninfer::ops::detail
