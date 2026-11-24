#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

namespace ninfer::models::qwen3_5::execution {

inline void project(const Tensor& input, const LinearParameters& p, Tensor& output,
                    MemoryArena& workspace, CudaExecutionView execution) {
    auto scope = workspace.scope();
    ops::linear(input, p.weight, output, p.policy, workspace, execution);
}

inline void project_add(const Tensor& input, const LinearParameters& p, Tensor& residual,
                        MemoryArena& workspace, CudaExecutionView execution) {
    auto scope = workspace.scope();
    ops::linear_add(input, p.weight, residual, p.policy, workspace, execution);
}

inline void project_swiglu(const Tensor& input, const LinearParameters& p, Tensor& output,
                           MemoryArena& workspace, CudaExecutionView execution) {
    auto scope = workspace.scope();
    ops::linear_swiglu(input, p.weight, output, p.policy, workspace, execution);
}

} // namespace ninfer::models::qwen3_5::execution
