#pragma once

#include "ops/linear/split/operands.h"

#include <array>

namespace ninfer::ops::detail::linear_split {
using SplitLaunch   = void (*)(const Tensor&, Weights, std::span<const Tensor>, MemoryArena*,
                             cudaStream_t);
using SplitCapacity = std::size_t (*)(LinearPolicy, std::int32_t, std::int32_t);

struct SplitShape {
    int first_rows, input_rows;
    std::span<const int> segments;
    SplitLaunch launch;
    SplitCapacity capacity = nullptr;
    int second_rows        = 0;
};

using SplitBiasLaunch = void (*)(const Tensor&, const Weight&, const Tensor&,
                                 std::span<const Tensor>, LinearPolicy, MemoryArena*, cudaStream_t);

struct SplitBiasShape {
    int rows, input_rows;
    std::span<const int> segments;
    SplitBiasLaunch launch;
    SplitCapacity capacity                                = nullptr;
    bool (*uses_tensor_cores)(LinearPolicy, std::int32_t) = nullptr;
};

template <int... Rows>
inline constexpr std::array kSplitRows{Rows...};

extern const SplitShape kBf16SplitN20480K1536;
extern const SplitShape kBf16SplitN3072K4096;
extern const SplitShape kBf16SplitN1152K2560, kBf16SplitN7296K2560;
extern const SplitShape kBf16SplitN1152K2560ThreeOutput, kBf16SplitN640K2560;
extern const SplitShape kBf16SplitN288K4096, kBf16SplitN1664K2560, kBf16SplitN13952K2560,
    kBf16SplitN14336K5120;
extern const SplitShape kQ8SplitN12288K2560, kQ8SplitN16384K2560, kQ8SplitN9216K2048;
extern const SplitShape kQ8SplitN6144K2048, kQ8SplitN6144K5120, kQ8SplitN12288K2048;
extern const SplitShape kQ8SplitN8192K2560, kQ8SplitN12288K4096;
extern const SplitShape kQ8SplitN6144K2560;
extern const SplitShape kQ4Q5SplitN7168N7168K5120, kQ4Q5SplitN4096N12288K5120;
extern const SplitShape kFp8SplitN14336K5120, kFp8SplitN16384K5120;
extern const SplitShape kNvfp4SplitN14336K5120, kNvfp4SplitN16384K5120;

extern const SplitShape kMxfp8SplitN20480K1536;
extern const SplitShape kFp8BlockBf16SplitN16384N4096K1536;
extern const SplitBiasShape kBf16SplitBiasN3072K1024, kMxfp8SplitBiasN3072K1024;

const SplitBiasShape& select_split_bias_shape(QType type, int n, int k, LinearPolicy policy,
                                              std::span<const std::int32_t> output_rows);
const SplitBiasShape& select_split_bias_shape(QType type, int n, int k, LinearPolicy policy,
                                              std::span<const Tensor> outputs);

const SplitShape& select_split_shape(ProjectionProfile weights,
                                     std::span<const std::int32_t> output_rows);
const SplitShape& select_split_shape(ProjectionProfile weights, std::span<const Tensor> outputs);
} // namespace ninfer::ops::detail::linear_split
