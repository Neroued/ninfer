#pragma once

#include "ops/linear/q6/q6_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Q6Launch select_q6_n248320_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n34816_k5120(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n248320_k2048(std::int32_t tokens);
[[nodiscard]] Q6Launch select_q6_n1152_k1536(std::int32_t tokens);

} // namespace ninfer::ops::detail
