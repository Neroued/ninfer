#include "ops/linear/projection/mxfp8/dispatch.h"
#include "ops/common/host/operands.h"
#include "core/platform.h"
#include "ops/linear/common/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const Mxfp8LinearShape& shape(int n, int k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("mxfp8 linear: invalid policy");
#if NINFER_TARGET_SPARK
    const Mxfp8LinearShape* shapes[] = {
        &kMxfp8N12288K4096, &kMxfp8N4096K4096,  &kMxfp8N4096K6144,  &kMxfp8N77440K4096,
        &kMxfp8N2048K4096,  &kMxfp8N4096K1024,  &kMxfp8N2208K4096,  &kMxfp8N2176K4096,
        &kMxfp8N640K4096,   &kMxfp8N20480K1536, &kMxfp8N16384K1536, &kMxfp8N4096K16384,
        &kMxfp8N3072K1024,  &kMxfp8N1024K1024,
    };
    for (const auto* shape : shapes)
        if (shape->n == n && shape->k == k) return *shape;
#endif
    throw std::invalid_argument("mxfp8 linear: unsupported shape");
}
} // namespace

const Mxfp8LinearRoute& select_mxfp8_linear_route(int n, int k, int tokens, LinearPolicy policy) {
    const auto& selected = shape(n, k, policy);
    if (tokens <= 0) throw std::invalid_argument("MXFP8 linear: tokens must be positive");
    const auto routes = allows_a8(policy) ? selected.a8 : selected.a16;
    for (const auto& route : routes)
        if (tokens <= route.last_tokens) return route;
    throw std::invalid_argument("MXFP8 linear: unsupported token count");
}

std::size_t mxfp8_linear_workspace_capacity_bytes(int n, int k, LinearPolicy policy, int min_tokens,
                                                  int max_tokens) {
    const auto& selected = shape(n, k, policy);
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("mxfp8 linear: invalid token interval");
    const auto routes = allows_a8(policy) ? selected.a8 : selected.a16;
    std::size_t peak  = 0;
    int first         = 1;
    for (const auto& route : routes) {
        const int last = std::min(route.last_tokens, max_tokens);
        if (route.compute == Mxfp8LinearCompute::A8 && first <= max_tokens && last >= min_tokens)
            peak = std::max(peak, mxfp8_a8_workspace_capacity_bytes(last, k, route.code_tokens(last)));
        if (route.last_tokens >= max_tokens) break;
        first = route.last_tokens + 1;
    }
    return peak;
}

void mxfp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    MemoryArena* workspace, cudaStream_t stream) {
    (void)common::validate_mxfp8_weight(weight, "mxfp8 linear");
    const auto& route = select_mxfp8_linear_route(weight.n, weight.k, x.ne[1], policy);
    if (route.compute != Mxfp8LinearCompute::A8) return route.launch(x, weight, out, {}, stream);
    if (!workspace) throw std::invalid_argument("MXFP8 A8 requires caller workspace");
    auto scope = workspace->scope();
    const auto backing =
        workspace->alloc_bytes(
            mxfp8_a8_workspace_capacity_bytes(x.ne[1], weight.k, route.code_tokens(x.ne[1])), 256);
    validate_projection_scratch(x, weight, {&out, 1}, nullptr, backing, "MXFP8 linear");
    MemoryArena local(backing);
    const auto scratch =
        allocate_mxfp8_a8_workspace(local, x.ne[1], weight.k, route.code_tokens(x.ne[1]));
    route.launch(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
