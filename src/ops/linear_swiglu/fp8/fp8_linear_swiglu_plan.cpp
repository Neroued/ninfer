#include "core/weight.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"

#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_config.h"

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

Fp8LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 linear_swiglu: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8LinearSwiGluRoute::A16; }
    if (!allows_a8(policy)) {
        throw std::invalid_argument("fp8 linear_swiglu admits only A16 or A8");
    }
    return tokens == 1 || tokens >= 3 ? Fp8LinearSwiGluRoute::A8 : Fp8LinearSwiGluRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kOutputRows = Fp8N34816K5120::kOutputRows / 2;
    constexpr std::int32_t kChunk      = 4;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * kOutputRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {kOutputRows, active});
        if (active == 1) {
            fp8_linear_swiglu_decode_launch(input_chunk, weight, output_chunk, stream);
        } else {
            fp8_linear_swiglu_small_t_launch(input_chunk, weight, output_chunk, stream);
        }
    }
}

} // namespace

std::size_t fp8_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                       std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    const bool interval_uses_a8 = allows_a8(policy) && (min_tokens == 1 || max_tokens >= 3);
    return interval_uses_a8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8N34816K5120::kInputRows)
               : 0;
}

void fp8_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8LinearSwiGluRoute::A16) {
        launch_a16(x, weight, out, stream);
        return;
    }
    fp8_linear_swiglu_a8_launch(x, weight, out, workspace, stream);
}

} // namespace ninfer::ops::detail
