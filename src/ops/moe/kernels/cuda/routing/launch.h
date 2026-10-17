#pragma once

#include "ops/moe/kernels/cuda/launch.h"

namespace ninfer::ops::detail::moe_cuda::routing {

enum class RouterAlgorithm : std::uint8_t { Vector, TokenTile, Mma };

struct RouterPlan {
    RouterAlgorithm algorithm = RouterAlgorithm::Vector;
    bool programmatic         = false;
    bool count_assignments    = false;
    int mma_rows              = 16;
    bool network_topk         = false;
};

struct RouteStorage {
    float* scores    = nullptr;
    int* local_ranks = nullptr;
    int* tile_counts = nullptr;
};

void launch_route(const Tensor& x, const Weight& control, const ExpertRoutes& routes,
                  const Tensor& shared, const RouterPlan& plan, const RouteStorage& storage,
                  const MoeExecutionHints& hints, CudaExecutionView execution);

template <class Arena>
RouteStorage allocate_route(Arena& arena, const RouterPlan& plan, int tokens, int experts) {
    const int rows =
        plan.algorithm == RouterAlgorithm::TokenTile ? (experts + 1) * 4 : (experts + 4) / 4 * 4;
    auto scores = arena.alloc(DType::FP32, {rows, tokens});
    return {static_cast<float*>(scores.data)};
}

} // namespace ninfer::ops::detail::moe_cuda::routing
