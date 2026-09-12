#include "ops/attn_input_proj/fp8/fp8_attn_input_plan.h"

#include "ops/linear/fp8/fp8_config.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Fp8AttnInputRoute : std::uint8_t {
    A16,
    A8,
};

Fp8AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("fp8 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Fp8AttnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("fp8 attn_input_proj: unsupported policy");
    }
    return tokens >= 5 ? Fp8AttnInputRoute::A8 : Fp8AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1)
        fp8_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= kFp8AttnInputLastSimtT)
        fp8_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
    else if (x.ne[1] <= kFp8AttnInputLastSmallMmaT)
        fp8_attn_input_a16_small_mma_launch(x, weight, q, gate, k, v, stream);
    else
        fp8_attn_input_a16_gemm_launch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t fp8_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("fp8 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Fp8AttnInputRoute::A8
               ? fp8_a8_workspace_capacity_bytes(max_tokens, Fp8AttnInputGeometry::kInputRows)
               : 0;
}

void fp8_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                             Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                             cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Fp8AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("fp8 A8 attn_input_proj requires caller workspace");
    }
    auto scope                   = workspace->scope();
    const Fp8A8Workspace scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
    fp8_attn_input_a8_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
