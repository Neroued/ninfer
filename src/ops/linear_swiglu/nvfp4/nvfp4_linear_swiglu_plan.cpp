#include "core/weight.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedW4A4,
    LinearW4A4Post,
    TmaFusedW4A4,
};

constexpr std::int32_t kFusedMaxTokens = 128;
constexpr std::int32_t kRaggedFloor    = kNvfp4TmaBlockM + kNvfp4TmaBlockM / 2;

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
    if (tokens <= kFusedMaxTokens) { return Nvfp4LinearSwiGluRoute::FusedW4A4; }
    // This route dispatches its own fused kernel rather than a Linear shape's, so it carries its
    // own condition; the call site below forces the matching scale layout.
    //
    // It no longer needs a whole number of tiles, but a width just past one tile would pay for a
    // second tile that is almost all padding - at 257 tokens that is a third of the work for one
    // token of it. So: one whole tile, or one and a half and up.
    if (tokens == kNvfp4TmaBlockM || tokens >= kRaggedFloor) {
        return Nvfp4LinearSwiGluRoute::TmaFusedW4A4;
    }
    return Nvfp4LinearSwiGluRoute::LinearW4A4Post;
}

struct Nvfp4LinearSwiGluWorkspace {
    Tensor projected;
    DeviceSpan linear;
};

template <class Allocator>
Nvfp4LinearSwiGluWorkspace allocate_baseline_workspace(Allocator& allocator, std::int32_t tokens) {
    Nvfp4LinearSwiGluWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {Nvfp4N34816K5120::kOutputRows, tokens}, 256);
    const std::size_t linear_bytes = linear_workspace_capacity_bytes(
        QType::NVFP4, Nvfp4N34816K5120::kOutputRows, Nvfp4N34816K5120::kInputRows,
        LinearPolicy::AllowA4, tokens, tokens);
    out.linear = allocator.alloc_bytes(linear_bytes, 256);
    return out;
}

template <class Allocator>
Nvfp4W4a4Workspace allocate_fused_workspace(Allocator& allocator, std::int32_t tokens) {
    return allocate_nvfp4_w4a4_workspace(allocator, tokens, Nvfp4N34816K5120::kInputRows);
}

std::size_t baseline_workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_baseline_workspace(layout, tokens);
    return layout.peak_bytes(1);
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

    std::size_t maximum = 0;
    if (min_tokens <= kFusedMaxTokens && max_tokens >= 5) {
        maximum = fused_workspace_bytes(std::min(max_tokens, kFusedMaxTokens));
    }
    // The fused TMA route is no longer confined to multiples of the tile, so the widest width that
    // reaches it is max_tokens itself whenever the route admits it.
    for (std::int32_t tokens = max_tokens; tokens >= std::max(min_tokens, kNvfp4TmaBlockM);
         --tokens) {
        if (resolve_route(policy, tokens) == Nvfp4LinearSwiGluRoute::TmaFusedW4A4) {
            maximum = std::max(maximum, fused_workspace_bytes(tokens));
            break;
        }
    }

    // Likewise the widest width that still falls through to the baseline route: scan down for it
    // rather than assume it is one below max_tokens.
    for (std::int32_t tokens = max_tokens; tokens >= std::max(min_tokens, kFusedMaxTokens + 1);
         --tokens) {
        if (resolve_route(policy, tokens) == Nvfp4LinearSwiGluRoute::LinearW4A4Post) {
            maximum = std::max(maximum, baseline_workspace_bytes(tokens));
            break;
        }
    }
    return maximum;
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
    case Nvfp4LinearSwiGluRoute::FusedW4A4:
        nvfp4_linear_swiglu_w4a4_launch(x, weight, out, workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::TmaFusedW4A4: {
        auto scope                       = workspace.scope();
        const Nvfp4W4a4Workspace scratch = allocate_fused_workspace(workspace, x.ne[1]);
        launch_nvfp4_w4a4_quantize(x, weight, scratch, Nvfp4ScaleLayout::Tiled, stream);
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_linear_swiglu_w4a4_tma(
            scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            x.ne[1], alpha, stream);
        return;
    }
    case Nvfp4LinearSwiGluRoute::LinearW4A4Post:
        break;
    }

    auto scope                         = workspace.scope();
    Nvfp4LinearSwiGluWorkspace scratch = allocate_baseline_workspace(workspace, x.ne[1]);
    WorkspaceArena linear_workspace(scratch.linear);
    linear(x, weight, scratch.projected, LinearPolicy::AllowA4, linear_workspace, stream);
    constexpr std::int32_t kIntermediate = Nvfp4N34816K5120::kOutputRows / 2;
    silu_mul(scratch.projected.slice(0, 0, kIntermediate),
             scratch.projected.slice(0, kIntermediate, kIntermediate), out, stream);
}

} // namespace ninfer::ops::detail
