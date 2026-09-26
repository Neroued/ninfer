#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_layout.h"
#include "ops/linear/nvfp4/nvfp4_a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_a4_tma_launch.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedA4,
    TmaFusedA4,
};

Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (!valid_linear_policy(policy)) {
        throw std::invalid_argument("nvfp4 linear_swiglu: invalid compute policy");
    }
    if (policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) {
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        if (tokens <= 16) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
        throw std::invalid_argument("nvfp4 linear_swiglu A16 is registered only through T=16");
    }
    if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
    if (tokens <= 4) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
    // This route dispatches its own fused kernel rather than a Linear shape's, so it carries its
    // own condition; the call site below forces the matching scale layout.
    if (tokens >= 256) { return Nvfp4LinearSwiGluRoute::TmaFusedA4; }
    return Nvfp4LinearSwiGluRoute::FusedA4;
}

template <class Allocator>
Nvfp4A4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_a4_workspace(allocator, tokens, Nvfp4N34816K5120::kInputRows);
}

std::size_t fused_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_fused_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

} // namespace

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    if ((policy == LinearPolicy::A16Only || policy == LinearPolicy::AllowA8) || max_tokens <= 4) {
        return 0;
    }

    return fused_workspace_bytes(max_tokens);
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedA4:
        nvfp4_linear_swiglu_a4_launch(x, weight, out, workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedA4: {
        auto scope                     = workspace.scope();
        const Nvfp4A4Workspace scratch = allocate_fused_workspace(workspace, x.ne[1]);
        launch_nvfp4_a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled256, stream);
        launch_nvfp4_linear_swiglu_a4_tma(
            nvfp4_a4_operands(weight, scratch, x.ne[1], Nvfp4ScaleLayout::Tiled256),
            static_cast<__nv_bfloat16*>(out.data), stream);
        return;
    }
    }
    throw std::logic_error("unreachable NVFP4 SwiGLU route");
}
} // namespace ninfer::ops::detail
