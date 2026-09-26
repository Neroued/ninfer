#include "core/weight.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_layout.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    A16,
    A4,
};

Nvfp4GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        return Nvfp4GdnInputRoute::A16;
    }
    if (allows_a4(policy)) { return Nvfp4GdnInputRoute::A4; }
    throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
}


} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Nvfp4GdnInputRoute::A4
               ? nvfp4_a4_workspace_capacity_bytes(max_tokens, Nvfp4N16384K5120::kInputRows)
               : 0;
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4GdnInputRoute::A16) {
        nvfp4_gdn_input_a16_launch(x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 A4 gdn_input_proj requires caller workspace");
    }
    auto scope                     = workspace->scope();
    const Nvfp4A4Workspace scratch = allocate_nvfp4_a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_a4_launch(x, weight, qkv, z, scratch, stream);
}

} // namespace ninfer::ops::detail
