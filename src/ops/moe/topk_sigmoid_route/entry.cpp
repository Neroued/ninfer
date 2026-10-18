#include "core/platform.h"
#include "core/layout.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/operands.h"
#include "ops/moe/topk_sigmoid_route/dispatch.h"
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
void validate(const SigmoidRouteProfile& p) {
#if NINFER_TARGET_SPARK
    if (p.route.hidden != 4096 || p.route.experts != 288 || p.route.top_k != 8 ||
        !std::isfinite(p.scaling_factor) || p.scaling_factor <= 0 ||
        !std::isfinite(p.normalization_epsilon) || p.normalization_epsilon <= 0)
        throw std::invalid_argument("topk_sigmoid_route: unsupported profile or normalization");
#else
    throw std::invalid_argument("topk_sigmoid_route: no registered profile on this target");
#endif
}
} // namespace

std::size_t topk_sigmoid_route_workspace_capacity_bytes(const SigmoidRouteProfile& p, int first,
                                                        int last) {
    validate(p);
    detail::validate_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP32, {p.route.experts, last});
    return layout.peak_bytes(1);
}

void topk_sigmoid_route(const Tensor& x, const SigmoidRouteProfile& p, const Weight& control,
                        const Tensor& bias, ExpertRoutes& routes, MemoryArena& ws,
                        CudaExecutionView execution) {
    validate(p);
    detail::MoeOperands operands("topk_sigmoid_route", execution);
    const int tokens = operands.input(x, p.route.hidden);
    operands.sigmoid_control(control, p.route);
    operands.coefficient(bias, p.route.experts, false);
    operands.routes(routes, p.route.top_k, tokens, true);
    operands.workspace(ws, topk_sigmoid_route_workspace_capacity_bytes(p, tokens, tokens));
    operands.finish();
#if NINFER_TARGET_SPARK
    auto scope  = ws.scope();
    auto scores = ws.alloc(DType::FP32, {p.route.experts, tokens});
    detail::moe_sigmoid::launch(x, control, bias, routes, p, static_cast<float*>(scores.data),
                                execution.stream());
#endif
}
} // namespace ninfer::ops
