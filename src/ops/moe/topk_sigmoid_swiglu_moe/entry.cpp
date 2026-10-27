#include "core/platform.h"
#include "core/layout.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/operands.h"
#include "ops/moe/swiglu/dispatch.h"
#include "ops/moe/routed_swiglu/cuda/contribution.h"
#include "ops/moe/topk_sigmoid_swiglu_moe/dispatch.h"
#include "ops/moe/kernels/cuda/block_scaled/clamped/workspace.h"
#include "ops/moe/workspace.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {
namespace {
struct WholeStorage {
    ExpertRoutes routes;
    Tensor routed;
    Tensor valid_tokens;
};

template <class Arena>
WholeStorage allocate_whole(Arena& arena, int tokens) {
    WholeStorage s;
    s.routes       = {arena.alloc(DType::I32, {8, tokens}), arena.alloc(DType::FP32, {8, tokens})};
    s.routed       = arena.alloc(DType::FP32, {4096, tokens});
    s.valid_tokens = arena.alloc(DType::I32, {tokens});
    return s;
}

std::size_t routed_capacity(const RoutedSwiGLUProfile& p, ExpertSubsetKind subset, int first,
                            int last) {
#if NINFER_TARGET_SPARK
    if (p.ffn.intermediate == 1024) {
        detail::validate_expert_profile(p);
        detail::validate_interval(first, last);
        if (subset == ExpertSubsetKind::None) return 0;
        const auto& shape = detail::moe_sigmoid::nvfp4_h4096_i1024_e288_k8();
        return detail::interval_capacity(
            first, last, shape.allocation_boundaries, shape.slice_tokens, [&](int tokens) {
                WorkspaceLayoutBuilder layout;
                (void)detail::moe_cuda::block_scaled::clamped::allocate(
                    layout, shape.select(p.ffn, tokens), tokens, true, true);
                return layout.peak_bytes(1);
            });
    }
#endif
    return detail::moe_routed::contribution_workspace_capacity_bytes(p, subset, first, last);
}

#if NINFER_TARGET_SPARK
detail::RoutedSwiGLUContribution
routed_contribution(const Tensor& x, const RoutedSwiGLUProfile& p, const CudaSwiGLUExperts& experts,
                    const ExpertRoutes& routes, ExpertSubset subset, WholeStorage& storage,
                    MemoryArena& scratch, CudaExecutionView execution) {
    auto* validity = static_cast<int*>(storage.valid_tokens.data);
    if (p.ffn.intermediate != 1024)
        return detail::moe_routed::execute_contribution(
            x, p, experts, routes, subset, storage.routed, validity, scratch, execution);
    const auto& shape = detail::moe_sigmoid::nvfp4_h4096_i1024_e288_k8();
    for (int start = 0; start < x.ne[1];) {
        const int tokens = std::min(x.ne[1] - start, shape.slice_tokens);
        const auto input = x.slice(1, start, tokens);
        auto output      = storage.routed.slice(1, start, tokens);
        const ExpertRoutes selected{routes.ids.slice(1, start, tokens),
                                    routes.coefficients.slice(1, start, tokens)};
        auto scope      = scratch.scope();
        const auto plan = shape.select(p.ffn, tokens);
        auto buffers =
            detail::moe_cuda::block_scaled::clamped::allocate(scratch, plan, tokens, true, true);
        buffers.active_tokens = validity + start;
        detail::moe_cuda::block_scaled::clamped::launch(
            input, experts, selected, subset, MoeEpilogue::Write, output, plan, buffers, execution);
        start += tokens;
    }
    return {static_cast<const float*>(storage.routed.data), validity};
}
#endif

void validate(const SigmoidMoeProfile& p, ExpertSubsetKind subset, int first, int last) {
    detail::validate_subset_kind(subset);
    if (p.routing.route.hidden != 4096 || p.routed.ffn.hidden != 4096 || p.shared.hidden != 4096 ||
        p.routed.experts != p.routing.route.experts || p.routed.top_k != p.routing.route.top_k ||
        p.shared.intermediate != 1024 ||
        (p.routed.ffn.gate_up_type != QType::NVFP4 &&
         p.routed.ffn.gate_up_type != QType::FP8_E4M3FN_BLOCK128X128_BF16))
        throw std::invalid_argument("topk_sigmoid_swiglu_moe: unsupported geometry or formats");
    (void)topk_sigmoid_route_workspace_capacity_bytes(p.routing, first, last);
    (void)routed_capacity(p.routed, subset, first, last);
    (void)swiglu_workspace_capacity_bytes(p.shared, first, last);
}
} // namespace

std::size_t topk_sigmoid_swiglu_moe_workspace_capacity_bytes(const SigmoidMoeProfile& p,
                                                             ExpertSubsetKind subset, int first,
                                                             int last) {
    validate(p, subset, first, last);
    if (subset == ExpertSubsetKind::None)
        return swiglu_workspace_capacity_bytes(p.shared, first, last);
    WorkspaceLayoutBuilder layout;
    (void)allocate_whole(layout, last);
    const auto work = std::max({topk_sigmoid_route_workspace_capacity_bytes(p.routing, first, last),
                                routed_capacity(p.routed, subset, first, last),
                                swiglu_workspace_capacity_bytes(p.shared, first, last)});
    (void)layout.alloc_bytes(work);
    return layout.peak_bytes(1);
}

void topk_sigmoid_swiglu_moe(const Tensor& x, const SigmoidMoeProfile& p,
                             const SigmoidMoeWeights& w, ExpertSubset subset, Tensor& out,
                             MemoryArena& ws, CudaExecutionView execution) {
    validate(p, subset.kind, 1, 1);
    detail::MoeOperands operands("topk_sigmoid_swiglu_moe", execution);
    const int tokens = operands.input(x, p.routed.ffn.hidden);
    if (out.dtype != DType::BF16)
        throw std::invalid_argument("whole local MoE requires BF16 Write");
    operands.output(out, p.routed.ffn.hidden, tokens, MoeEpilogue::Write);
    operands.unscaled_shared(w.shared, p.shared);
    operands.subset(subset, p.routed.experts);
    if (subset.kind != ExpertSubsetKind::None) {
        operands.sigmoid_control(w.control, p.routing.route);
        operands.coefficient(w.correction_bias, p.routing.route.experts, false);
        operands.experts(w.routed, p.routed);
    }
    auto scope = ws.scope();
    MemoryArena scratch(ws.alloc_bytes(
        topk_sigmoid_swiglu_moe_workspace_capacity_bytes(p, subset.kind, tokens, tokens)));
    operands.workspace(scratch.storage());
    operands.finish();
    if (subset.kind == ExpertSubsetKind::None) {
        swiglu(x, p.shared, w.shared, out, scratch, execution);
        return;
    }
#if NINFER_TARGET_SPARK
    auto storage = allocate_whole(scratch, tokens);
    topk_sigmoid_route(x, p.routing, w.control, w.correction_bias, storage.routes, scratch,
                       execution);
    const auto routed = routed_contribution(x, p.routed, w.routed, storage.routes, subset, storage,
                                            scratch, execution);
    detail::swiglu_ffn::launch_a16_ffn_join(x, w.shared, out, routed, *p.shared.clamp_limit,
                                            scratch, execution.stream());
#endif
}
} // namespace ninfer::ops
