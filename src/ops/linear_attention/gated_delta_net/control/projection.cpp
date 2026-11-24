#include "ninfer/ops/gdn_control.h"
#include "ops/linear_attention/gated_delta_net/control/execute.h"
#include "ops/linear_attention/gated_delta_net/control/bf16/launch.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops {
namespace control = detail::gdn_control;

std::size_t gdn_control_proj_workspace_capacity_bytes(const ProjectionProfile& weights,
                                                      std::int32_t first, std::int32_t last) {
    return control::capacity(control::describe_profile(weights), false, first, last);
}

std::size_t gdn_norm_control_proj_workspace_capacity_bytes(const ProjectionProfile& weights,
                                                           std::int32_t first, std::int32_t last) {
    return control::capacity(control::describe_profile(weights), true, first, last);
}

void gdn_control_proj(const Tensor& x, const ProjectionWeights& weights, const Tensor& a_log,
                      const Tensor& dt_bias, GdnControlOutputs out, MemoryArena& workspace,
                      CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("gdn_control_proj: workspace belongs to another device");
    }
    const auto operands = control::describe_weights(weights);
    control::validate_weights(operands);
    control::validate_projection(x, operands, a_log, dt_bias, out);
    const auto plan    = control::select_projection_plan(operands.geometry, x.ne[1]);
    auto scope         = workspace.scope();
    const auto scratch = control::allocate_workspace(workspace, operands.geometry, plan, x.ne[1]);
    const std::array<const Tensor*, 5> tensors{&x, &a_log, &dt_bias, &out.g, &out.beta};
    control::validate_storage(&operands, tensors, execution, scratch.partial);
    control::execute_projection(x, operands, a_log, dt_bias, out, plan, scratch, execution);
}

void gdn_norm_control_proj(const Tensor& x, const Tensor& norm_weight, float eps,
                           const ProjectionWeights& weights, const Tensor& a_log,
                           const Tensor& dt_bias, GdnNormControlOutputs out, MemoryArena& workspace,
                           CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument("gdn_norm_control_proj: workspace belongs to another device");
    }
    const auto operands = control::describe_weights(weights);
    control::validate_weights(operands);
    control::validate_projection(x, operands, a_log, dt_bias, out.control);
    control::validate_norm(x, norm_weight, eps, operands.geometry, out.normalized);
    const auto plan = control::select_norm_plan(operands.geometry, x.ne[1]);
    auto scope      = workspace.scope();
    const auto scratch =
        control::allocate_workspace(workspace, operands.geometry, plan.projection, x.ne[1],
                                    plan.kernel == control::NormKernel::Mma);
    const std::array<const Tensor*, 7> tensors{
        &x, &norm_weight, &a_log, &dt_bias, &out.normalized, &out.control.g, &out.control.beta};
    control::validate_storage(&operands, tensors, execution, scratch.partial);
    control::execute_norm(x, norm_weight, eps, operands, a_log, dt_bias, out, plan, scratch,
                          execution);
}

} // namespace ninfer::ops
