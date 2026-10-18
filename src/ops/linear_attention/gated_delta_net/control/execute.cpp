#include "ops/linear_attention/gated_delta_net/control/execute.h"
#include "ops/linear_attention/gated_delta_net/control/bf16/launch.h"
#include "ninfer/ops/rmsnorm.h"
#include "core/platform.h"

namespace ninfer::ops::detail::gdn_control {
namespace {

Inputs inputs(const Tensor& x, const Weights& weights, const Tensor& a_log, const Tensor& dt_bias) {
    const void* a = weights.first->qdata;
    const void* b = weights.second
                        ? weights.second->qdata
                        : static_cast<const std::byte*>(a) +
                              std::size_t(weights.geometry.heads) * weights.geometry.input_rows * 2;
    return {x, a_log, dt_bias, a, b};
}

} // namespace

void execute_projection(const Tensor& x, const Weights& weights, const Tensor& a_log,
                        const Tensor& dt_bias, GdnControlOutputs out, Plan plan, Scratch scratch,
                        CudaExecutionView execution) {
    const auto in = inputs(x, weights, a_log, dt_bias);
    switch (weights.geometry.input_rows) {
    case 5120:
        launch_n96_k5120(in, out, plan, scratch, execution);
        return;
    case 2048:
        launch_n64_k2048(in, out, plan, scratch, execution);
        return;
    case 2560:
#if NINFER_TARGET_SPARK
        if (weights.geometry.heads == 24) {
            launch_n48_k2560(in, out, plan, scratch, execution);
            return;
        }
#endif
        launch_n96_k2560(in, out, plan, scratch, execution);
        return;
    }
    throw std::logic_error("gdn_control: missing geometry launcher");
}

void execute_norm(const Tensor& x, const Tensor& norm_weight, float eps, const Weights& weights,
                  const Tensor& a_log, const Tensor& dt_bias, GdnNormControlOutputs out,
                  NormPlan plan, Scratch scratch, CudaExecutionView execution) {
    const auto in = inputs(x, weights, a_log, dt_bias);
    const Norm norm{norm_weight, eps, out.normalized};
    if (plan.kernel == NormKernel::Simt) {
        launch_norm_n96_k5120(in, norm, out.control, plan, execution.stream());
        return;
    }
    if (plan.kernel == NormKernel::Mma &&
        launch_norm_n64_k2048(in, norm, out.control, plan, scratch, execution))
        return;
    rmsnorm(x, norm_weight, eps, true, out.normalized, execution);
    const auto control = plan.kernel == NormKernel::Materialized ? plan.projection : plan.fallback;
    execute_projection(out.normalized, weights, a_log, dt_bias, out.control, control, scratch,
                       execution);
}

} // namespace ninfer::ops::detail::gdn_control
