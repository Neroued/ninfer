#include "models/inkling2/execution/moe.h"

#include "models/inkling2/execution/workspace.h"

namespace ninfer::models::inkling2::execution {

void routed_moe(const Parameters& p, const LogicalLayerConfig& logical, const Tensor& input,
                ops::MoeTokenValidity validity, Tensor& output, MemoryArena& arena,
                CudaExecutionView cuda) {
    const auto& layer   = p.text.layers.at(logical.physical_layer);
    const auto& weights = layer.moe_passes.at(logical.pass_index);
    const auto& profile =
        p.profile->text.layers.at(logical.physical_layer).moe_passes.at(logical.pass_index);
    auto scope   = arena.scope();
    auto buffers = workspace::moe(arena, p.profile->config.text, input.ne[1]);
    ops::ExpertRoutes routes{buffers.route_ids, buffers.route_weights};
    ops::topk_sigmoid_route(input, profile.routing, weights.router, layer.correction_bias, routes,
                            validity, arena, cuda);
    ops::routed_swiglu(input, profile.routed, weights.routed, routes, weights.subset,
                       ops::MoeEpilogue::Write, output, arena, cuda);
}

} // namespace ninfer::models::inkling2::execution
