#include "core/platform.h"
#include "core/layout.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/operands.h"
#include "ops/moe/swiglu/dispatch.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {
namespace {
struct WholeStorage {
    ExpertRoutes routes;
    Tensor routed;
};

template <class Arena>
WholeStorage allocate_whole(Arena& arena, int tokens) {
    WholeStorage s;
    s.routes = {arena.alloc(DType::I32, {8, tokens}), arena.alloc(DType::FP32, {8, tokens})};
    s.routed = arena.alloc(DType::FP32, {4096, tokens});
    return s;
}

void validate(const SigmoidMoeProfile& p, ExpertSubsetKind subset, int first, int last) {
    detail::validate_subset_kind(subset);
    if (p.routing.route.hidden != 4096 || p.routed.ffn.hidden != 4096 || p.shared.hidden != 4096 ||
        p.routed.experts != p.routing.route.experts || p.routed.top_k != p.routing.route.top_k ||
        p.shared.intermediate != 1024 ||
        (p.routed.ffn.gate_up_type != QType::NVFP4 &&
         p.routed.ffn.gate_up_type != QType::FP8_E4M3FN_BLOCK128X128_BF16))
        throw std::invalid_argument("topk_sigmoid_swiglu_moe: unsupported geometry or formats");
    (void)topk_sigmoid_route_workspace_capacity_bytes(p.routing, first, last);
    (void)routed_swiglu_workspace_capacity_bytes(p.routed, subset, DType::FP32, MoeEpilogue::Write,
                                                 first, last);
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
                                routed_swiglu_workspace_capacity_bytes(
                                    p.routed, subset, DType::FP32, MoeEpilogue::Write, first, last),
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
    operands.workspace(
        ws, topk_sigmoid_swiglu_moe_workspace_capacity_bytes(p, subset.kind, tokens, tokens));
    operands.finish();
    if (subset.kind == ExpertSubsetKind::None) {
        swiglu(x, p.shared, w.shared, out, ws, execution);
        return;
    }
#if NINFER_TARGET_SPARK
    auto scope   = ws.scope();
    auto storage = allocate_whole(ws, tokens);
    topk_sigmoid_route(x, p.routing, w.control, w.correction_bias, storage.routes, ws, execution);
    routed_swiglu(x, p.routed, w.routed, storage.routes, subset, MoeEpilogue::Write, storage.routed,
                  ws, execution);
    detail::swiglu_ffn::launch_a16_ffn_join(x, w.shared, out,
                                            static_cast<const float*>(storage.routed.data),
                                            *p.shared.clamp_limit, ws, execution.stream());
#endif
}
} // namespace ninfer::ops
