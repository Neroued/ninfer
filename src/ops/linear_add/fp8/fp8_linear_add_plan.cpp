#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_geometry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8LinearAddRoute : std::uint8_t {
    A16,
    A8,
};

Fp8LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != Fp8N5120K6144::kOutputRows ||
        (input_rows != Fp8N5120K6144::kInputRows && input_rows != Fp8N5120K17408::kInputRows)) {
        throw std::invalid_argument("fp8 linear_add: unsupported shape");
    }
    if (policy == LinearPolicy::A16Only) { return Fp8LinearAddRoute::A16; }
    if (!allows_a8(policy)) { throw std::invalid_argument("fp8 linear_add: unsupported policy"); }
    const std::int32_t first_a8 = input_rows == Fp8N5120K6144::kInputRows ? 22 : 25;
    return tokens >= first_a8 ? Fp8LinearAddRoute::A8 : Fp8LinearAddRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    if (x.ne[1] == 1) return fp8_linear_add_decode_launch(x, weight, residual, stream);
    if (x.ne[1] <= 4) return fp8_linear_add_small_t_launch(x, weight, residual, stream);
    fp8_linear_add_matrix_launch(x, weight, residual, stream);
}

} // namespace

std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Fp8LinearAddRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    const Fp8LinearAddRoute route = resolve_route(weight.n, weight.k, policy, x.ne[1]);
    if (route == Fp8LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
    fp8_linear_add_a8_launch(x, weight, residual, workspace, stream);
}

} // namespace ninfer::ops::detail
