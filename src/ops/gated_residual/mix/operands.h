#pragma once

#include "ninfer/ops/gated_residual.h"
#include "ops/gated_residual/operands.h"

namespace ninfer::ops::detail::hc::mix {
inline constexpr int kWidth = 2560, kRank = 320;
inline constexpr int kResidualWidth = kWidth * kStreams;
enum class Kind { Block, Final };

constexpr int down_rows(Kind kind) { return kRank + (kind == Kind::Block ? kStreams : 0); }

struct Weights {
    Kind kind;
    const Weight& down;
    const Weight& up;
};

struct Operands {
    const void* normalized;
    const void* down;
    const void* up;
    void* output;
    float* injection;
    std::int32_t columns;
};

void describe_weights(Weights weights);
void validate_interval(std::int32_t first, std::int32_t last);
Operands validate(Weights weights, const Tensor& normalized, Tensor& output, Tensor* injection,
                  CudaExecutionView execution);
void validate_scratch(Weights weights, const Tensor& normalized, const Tensor& output,
                      const Tensor* injection, MemorySpan scratch);
} // namespace ninfer::ops::detail::hc::mix
