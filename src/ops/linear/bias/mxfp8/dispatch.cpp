#include "ops/linear/bias/mxfp8/dispatch.h"

#include "core/platform.h"
#include "ops/common/host/operands.h"
#include "ops/linear/common/operands.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

std::span<const Mxfp8BiasRoute> select_routes(int rows, int k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("MXFP8 linear_bias: invalid policy");
#if NINFER_TARGET_SPARK
    const auto& shape = kMxfp8BiasN1024K1024;
    if (rows == shape.rows && k == shape.k) return allows_a8(policy) ? shape.a8 : shape.a16;
#endif
    throw std::invalid_argument("MXFP8 linear_bias: unsupported weight geometry");
}

} // namespace

Mxfp8BiasRoute select_mxfp8_bias_route(int rows, int k, int tokens, LinearPolicy policy) {
    if (tokens <= 0) throw std::invalid_argument("MXFP8 linear_bias: tokens must be positive");
    for (const auto& route : select_routes(rows, k, policy))
        if (tokens <= route.last_tokens) return route;
    throw std::invalid_argument("MXFP8 linear_bias: unsupported token count");
}

std::size_t mxfp8_bias_workspace_capacity_bytes(int rows, int k, LinearPolicy policy, int first,
                                                int last) {
    if (first <= 0 || last < first)
        throw std::invalid_argument("MXFP8 linear_bias workspace: invalid token interval");
    std::size_t peak = 0;
    int begin        = 1;
    for (const auto& route : select_routes(rows, k, policy)) {
        const int end = std::min(route.last_tokens, last);
        if (route.quantize && begin <= last && end >= first)
            peak =
                std::max(peak, mxfp8_a8_workspace_capacity_bytes(end, k, route.code_tokens(end)));
        if (route.last_tokens >= last) return peak;
        begin = route.last_tokens + 1;
    }
    throw std::invalid_argument("MXFP8 linear_bias workspace: unsupported token interval");
}

void mxfp8_bias_dispatch(const Tensor& x, const Weight& weight, const Tensor& bias, Tensor& out,
                         LinearPolicy policy, MemoryArena* workspace, cudaStream_t stream) {
    (void)common::validate_mxfp8_weight(weight, "MXFP8 linear_bias");
    const auto route = select_mxfp8_bias_route(weight.n, weight.k, x.ne[1], policy);
    if (!route.quantize) return route.launch(x, weight, bias, out, {}, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("MXFP8 linear_bias A8 requires caller workspace");
    auto scope            = workspace->scope();
    const int code_tokens = route.code_tokens(x.ne[1]);
    const auto backing    = workspace->alloc_bytes(
        mxfp8_a8_workspace_capacity_bytes(x.ne[1], weight.k, code_tokens), 256);
    validate_projection_scratch(x, weight, {&out, 1}, &bias, backing, "MXFP8 linear_bias");
    MemoryArena local(backing);
    const auto scratch = allocate_mxfp8_a8_workspace(local, x.ne[1], weight.k, code_tokens);
    route.launch(x, weight, bias, out, scratch, stream);
}

} // namespace ninfer::ops::detail
