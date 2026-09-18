#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_shapes.h"
#include "ops/linear/fp8/fp8_format.h"
#include <array>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
const std::array kShapes{&kFp8N14336K5120, &kFp8N16384K5120, &kFp8N34816K5120,
                         &kFp8N5120K6144,  &kFp8N5120K17408, &kFp8N248320K5120};

const Fp8LinearShape& resolve_shape(std::int32_t n, std::int32_t k, LinearPolicy policy) {
    if (!valid_linear_policy(policy)) throw std::invalid_argument("fp8 linear: unsupported policy");
    for (const auto* shape : kShapes)
        if (shape->n == n && shape->k == k) return *shape;
    throw std::invalid_argument("fp8 linear: unsupported shape");
}
} // namespace

std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t n, std::int32_t k, LinearPolicy policy,
                                                std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens)
        throw std::invalid_argument("fp8 linear workspace: invalid token interval");
    const auto& shape = resolve_shape(n, k, policy);
    return allows_a8(policy) && shape.uses_a8(min_tokens, max_tokens)
               ? fp8_a8_workspace_capacity_bytes(max_tokens, k)
               : 0;
}

void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  WorkspaceArena* workspace, cudaStream_t stream) {
    validate_fp8_weight(weight, "fp8 linear");
    if (x.ne[1] <= 0) throw std::invalid_argument("fp8 linear: T must be positive");
    const auto& shape = resolve_shape(weight.n, weight.k, policy);
    if (!allows_a8(policy) || !shape.uses_a8(x.ne[1], x.ne[1]))
        return shape.a16(x, weight, out, stream);
    if (workspace == nullptr)
        throw std::invalid_argument("fp8 A8 linear requires caller workspace");
    auto scope         = workspace->scope();
    const auto scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
    shape.a8(x, weight, out, scratch, stream);
}
} // namespace ninfer::ops::detail
