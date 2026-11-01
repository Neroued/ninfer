#pragma once

#include "ninfer/ops/linear_split.h"

namespace ninfer::ops::detail::linear_split {
struct Weights {
    const Weight& first;
    const Weight* second = nullptr;
    LinearPolicy policy  = LinearPolicy::A16Only;
};
struct SplitShape;

Weights describe_weights(const ProjectionWeights& weights);
void validate_interval(std::int32_t first, std::int32_t last);
void validate_operands(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
                       const SplitShape& shape, CudaExecutionView execution);
void validate_scratch(const Tensor& x, Weights weights, std::span<const Tensor> outputs,
                      MemorySpan scratch);
} // namespace ninfer::ops::detail::linear_split
