#include "core/weight.h"
#include "ops/linear/swiglu/fp8/dispatch.h"

#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include "ops/linear/kernels/fp8/fp8_geometry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearSwiGluRoute : std::uint8_t {
    A16,
    A8,
};

Fp8LinearSwiGluRoute select_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 linear_swiglu: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8LinearSwiGluRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 linear_swiglu admits only A16 or A8");
    }
    return tokens >= 5 ? Fp8LinearSwiGluRoute::A8 : Fp8LinearSwiGluRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] <= 4) return launch_fp8_swiglu_small_t(x, weight, out, stream);
    launch_fp8_swiglu_matrix(x, weight, out, stream);
}

} // namespace

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_swiglu workspace: invalid token interval");
    }
    (void)select_route(policy, min_tokens);
    (void)select_route(policy, max_tokens);
    const bool interval_uses_a8 = allows_a8(policy) && max_tokens >= 5;
    return interval_uses_a8 ? fp8_a8_workspace_capacity_bytes(
                                  max_tokens, Fp8N34816K5120::kInputRows,
                                  fp8_linear_swiglu_partial_capacity_bytes(max_tokens))
                            : 0;
}

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, MemoryArena& workspace, cudaStream_t stream) {
    if (select_route(policy, x.ne[1]) == Fp8LinearSwiGluRoute::A16) {
        launch_a16(x, weight, out, stream);
        return;
    }
    launch_fp8_swiglu_a8(x, weight, out, workspace, stream);
}

} // namespace ninfer::ops::detail
