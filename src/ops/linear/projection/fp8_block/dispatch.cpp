#include "ops/linear/projection/fp8_block/dispatch.h"
#include "ops/common/host/operands.h"
#include "core/platform.h"
#include "ops/linear/common/operands.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const Fp8BlockLinearShape& shape(int n, int k, LinearPolicy policy) {
    if (!valid_linear_policy(policy))
        throw std::invalid_argument("fp8_block linear: invalid policy");
#if NINFER_TARGET_SPARK
    const Fp8BlockLinearShape* shapes[] = {
        &kFp8BlockN2048K4096,  &kFp8BlockN512K4096,  &kFp8BlockN16384K1536,
        &kFp8BlockN4096K16384, &kFp8BlockN4096K1024,
    };
    for (const auto* shape : shapes)
        if (shape->n == n && shape->k == k) return *shape;
#endif
    throw std::invalid_argument("fp8_block linear: unsupported shape");
}
} // namespace

std::size_t fp8_block_linear_workspace_capacity_bytes(int n, int k, LinearPolicy policy,
                                                      int min_tokens, int max_tokens) {
    const auto& selected = shape(n, k, policy);
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8_block linear: invalid token interval");
    return selected.workspace_capacity ? selected.workspace_capacity(min_tokens, max_tokens) : 0;
}

Fp8BlockLinearRoute select_fp8_block_linear_route(int n, int k, int tokens, LinearPolicy policy) {
    if (tokens <= 0) throw std::invalid_argument("FP8 block linear: tokens must be positive");
    return shape(n, k, policy).select(tokens, policy);
}

void fp8_block_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                        MemoryArena* workspace, cudaStream_t stream) {
    (void)common::validate_fp8_block_fp32_weight(weight, "fp8_block linear");
    const auto route = select_fp8_block_linear_route(weight.n, weight.k, x.ne[1], policy);
    if (workspace && route.workspace_launch) {
        auto scope         = workspace->scope();
        const auto backing = workspace->alloc_bytes(route.workspace_bytes, 1);
        validate_projection_scratch(x, weight, {&out, 1}, nullptr, backing, "fp8_block linear");
        return route.workspace_launch(x, weight, out, backing, stream);
    }
    route.launch(x, weight, out, stream);
}
} // namespace ninfer::ops::detail
