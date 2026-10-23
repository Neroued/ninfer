#pragma once

#include "ops/linear/kernels/bf16/bf16_launch.h"

namespace ninfer::ops::detail {
[[nodiscard]] Bf16Launch select_bf16_n8192_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k4096(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n4096_k4096(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n3072_k1024(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1024_k1176(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n4096_k8192(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n16384_k1536(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n20480_k1536(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1152_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n7296_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n640_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k320(std::int32_t tokens);


[[nodiscard]] Bf16Launch select_bf16_n77440_k4096(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n4096_k1024(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n2048_k4096(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n4096_k128(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n288_k4096(std::int32_t tokens);

[[nodiscard]] Bf16Launch select_bf16_n2560_k3072(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n124160_k2560(std::int32_t tokens);

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
