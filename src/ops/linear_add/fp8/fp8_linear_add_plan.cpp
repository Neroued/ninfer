#include "core/weight.h"
#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"

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
    for (std::int32_t token_begin = 0; token_begin < x.ne[1];
         token_begin += kFp8LinearAddChunkTokens) {
        const std::int32_t active = std::min(kFp8LinearAddChunkTokens, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(residual.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor residual_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            fp8_linear_add_decode_launch(input_chunk, weight, residual_chunk, stream);
        } else {
            fp8_linear_add_small_t_launch(input_chunk, weight, residual_chunk, stream);
        }
    }
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
