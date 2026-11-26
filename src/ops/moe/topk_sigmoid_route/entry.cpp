#include "ninfer/ops/moe.h"
#include "ops/moe/operands.h"
#include "ops/moe/topk_sigmoid_route/dispatch.h"
#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {
void validate(const SigmoidRouteProfile& p) {
    const auto& route = detail::moe_sigmoid::registered_route;
    if (p.route.hidden != route.hidden || p.route.experts != route.experts ||
        p.route.top_k != route.top_k || !std::isfinite(p.scaling_factor) || p.scaling_factor <= 0 ||
        !std::isfinite(p.normalization_epsilon) || p.normalization_epsilon <= 0)
        throw std::invalid_argument("topk_sigmoid_route: unsupported profile or normalization");
}
} // namespace

std::size_t topk_sigmoid_route_workspace_capacity_bytes(const SigmoidRouteProfile& p, int first,
                                                        int last) {
    validate(p);
    detail::validate_interval(first, last);
    return detail::moe_sigmoid::workspace_capacity_bytes(first, last);
}

void topk_sigmoid_route(const Tensor& x, const SigmoidRouteProfile& p, const Weight& control,
                        const Tensor& bias, ExpertRoutes& routes, MoeTokenValidity validity,
                        MemoryArena& ws, CudaExecutionView execution) {
    validate(p);
    detail::MoeOperands operands("topk_sigmoid_route", execution);
    const int tokens = operands.input(x, p.route.hidden);
    operands.validity(validity, tokens);
    operands.sigmoid_control(control, p.route);
    operands.coefficient(bias, p.route.experts, false);
    operands.routes(routes, p.route.top_k, tokens, true);
    auto scope    = ws.scope();
    float* logits = nullptr;
    if (const auto bytes = detail::moe_sigmoid::workspace_capacity_bytes(tokens, tokens)) {
        const auto scratch = ws.alloc_bytes(bytes);
        operands.workspace(scratch);
        logits = static_cast<float*>(scratch.data);
    }
    operands.finish();
    detail::moe_sigmoid::launch(x, control, bias, routes, p, logits,
                                detail::moe_cuda::token_validity(validity), execution.stream());
}
} // namespace ninfer::ops
