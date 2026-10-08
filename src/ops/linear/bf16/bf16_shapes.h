#pragma once

#include "ops/linear/bf16/bf16_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n48_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n96_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1664_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n324_k10240(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n320_k10240(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n10240_k320(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n12800_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n248320_k2560(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n13952_k2560(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n2560_k6144(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n2560_k2560(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n1152_k1536(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n3456_k1152(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n1152_k1152(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n4304_k1152(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1152_k4304(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n4608_k4608(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n2560_k4608(std::int32_t tokens);

} // namespace ninfer::ops::detail
