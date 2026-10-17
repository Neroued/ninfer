#pragma once

#include "core/device.h"
#include "core/tensor.h"

namespace ninfer::ops::detail::hc {
inline constexpr int kWidth = 2560, kStreams = 4, kRank = 320;
inline constexpr int kResidualWidth = kWidth * kStreams;

enum class Form { ExpandNorm, GroupedNorm, Combine, CombineNorm, AddNorm };

struct Operands {
    const void* input;
    const void* block;
    const float* injection;
    const void* weight;
    void* residual;
    void* normalized;
    std::int32_t columns;
    float epsilon;
};

Operands validate(Form form, const Tensor& input, const Tensor* block, const Tensor* injection,
                  const Tensor* weight, float epsilon, Tensor* residual, Tensor* normalized,
                  CudaExecutionView execution, const char* op);
void launch_norm_combine(Form form, Operands operands, CudaExecutionView execution);
} // namespace ninfer::ops::detail::hc
