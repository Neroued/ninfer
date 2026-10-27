#include "ops/gated_residual/mix/bf16/implementation.h"
#include "ops/gated_residual/mix/bf16/launch.h"

#include <stdexcept>

namespace ninfer::ops::detail::hc::mix::bf16 {

void execute(Weights weights, const Tensor& normalized, MemoryArena& workspace, Tensor& output,
             Tensor* injection, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("hc_mix: workspace belongs to another device");
    }
    const auto o = validate(weights, normalized, output, injection, execution);
    const auto p = select_plan(weights.kind, o.columns);
    auto scope   = workspace.scope();
    const auto scratch =
        allocate_workspace(workspace, layout(weights.kind, o.columns, split_factor(p.down)));
    validate_scratch(weights, normalized, output, injection, scratch.storage);
    launch(weights.kind, o, p, scratch, execution);
}
} // namespace ninfer::ops::detail::hc::mix::bf16
