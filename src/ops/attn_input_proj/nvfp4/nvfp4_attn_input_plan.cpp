#include "core/weight.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    A4,
};

Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4AttnInputRoute::A16;
    }
    if (!allows_a4(policy)) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    return tokens >= 4 ? Nvfp4AttnInputRoute::A4 : Nvfp4AttnInputRoute::A16;
}


} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Nvfp4AttnInputRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N14336K5120::kInputRows)
               : 0;
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        nvfp4_attn_input_a16_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 attn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
