#include "ops/linear/swiglu/mxfp8/dispatch.h"

#include "core/platform.h"
#include "ops/common/host/operands.h"
#include "ops/linear/common/operands.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

std::span<const Mxfp8SwiGluRoute> select_routes(int rows, int k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("MXFP8 linear_swiglu: invalid policy");
#if NINFER_TARGET_SPARK
    const std::array shapes{&kMxfp8SwiGluN12288K4096, &kMxfp8SwiGluN2048K4096};
    for (const auto* shape : shapes)
        if (rows == shape->rows && k == shape->k) return allows_a8(policy) ? shape->a8 : shape->a16;
#endif
    throw std::invalid_argument("MXFP8 linear_swiglu: unsupported weight geometry");
}

} // namespace

Mxfp8SwiGluRoute select_mxfp8_swiglu_route(int rows, int k, int tokens, LinearPolicy policy) {
    if (tokens <= 0) throw std::invalid_argument("MXFP8 linear_swiglu: tokens must be positive");
    for (const auto& route : select_routes(rows, k, policy))
        if (tokens <= route.last_tokens) return route;
    throw std::invalid_argument("MXFP8 linear_swiglu: unsupported token count");
}

std::size_t mxfp8_swiglu_workspace_capacity_bytes(int rows, int k, LinearPolicy policy, int first,
                                                  int last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("MXFP8 linear_swiglu workspace: invalid token interval");
    std::size_t peak = 0;
    int begin        = 1;
    for (const auto& route : select_routes(rows, k, policy)) {
        const int end = std::min(route.last_tokens, last);
        if (route.quantize && begin <= last && end >= first)
            peak = std::max(peak, mxfp8_a8_workspace_capacity_bytes(route.scale_tokens(end), k,
                                                                    route.code_tokens(end)));
        if (route.last_tokens >= last) return peak;
        begin = route.last_tokens + 1;
    }
    throw std::invalid_argument("MXFP8 linear_swiglu workspace: unsupported token interval");
}

void mxfp8_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out, float limit,
                           bool clamped, LinearPolicy policy, MemoryArena& workspace,
                           cudaStream_t stream) {
    (void)common::validate_mxfp8_weight(weight, "MXFP8 linear_swiglu");
    const auto route = select_mxfp8_swiglu_route(weight.n, weight.k, x.ne[1], policy);
    if (!route.quantize) return route.launch(x, weight, out, limit, clamped, {}, stream);
    auto scope            = workspace.scope();
    const int code_tokens = route.code_tokens(x.ne[1]);
    const auto backing    = workspace.alloc_bytes(
        mxfp8_a8_workspace_capacity_bytes(route.scale_tokens(x.ne[1]), weight.k, code_tokens), 256);
    validate_projection_scratch(x, weight, {&out, 1}, nullptr, backing, "MXFP8 linear_swiglu");
    MemoryArena local(backing);
    const auto scratch =
        allocate_mxfp8_a8_workspace(local, route.scale_tokens(x.ne[1]), weight.k, code_tokens);
    route.launch(x, weight, out, limit, clamped, scratch, stream);
}

} // namespace ninfer::ops::detail
