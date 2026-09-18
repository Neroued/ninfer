#pragma once

#include "ops/linear/q5/q5_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Q5Launch select_q5_n1024_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n6144_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n7168_k5120(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n5120_k17408(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n1152_k1152(std::int32_t tokens);
[[nodiscard]] Q5Launch select_q5_n1152_k4304(std::int32_t tokens);

} // namespace ninfer::ops::detail
