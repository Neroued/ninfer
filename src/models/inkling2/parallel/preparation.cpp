#include "models/inkling2/parallel/preparation.h"

#include <stdexcept>

namespace ninfer::models::inkling2 {

context::ContextDirectorySpec plan_group_context(const ParallelPlan& parallel,
                                                 std::span<const RankResourceDescription> ranks) {
    if (ranks.size() != 1 || parallel.ranks.size() != 1)
        throw std::invalid_argument("Inkling2 currently supports single-rank execution only");
    const auto& r = ranks.front();
    const auto& c = r.context;
    if (r.rank != 0 || c.ranks.size() != 1 || c.ranks.front().rank != 0 ||
        c.state.layouts.size() != 1 || c.pools.size() != r.caches.size()) {
        throw std::invalid_argument("Inkling2 rank resources disagree with single-rank layout");
    }
    return c;
}

PreparedContext prepare_context(const ParallelPlan& parallel,
                                std::span<const RankResourceDescription> ranks) {
    auto spec = plan_group_context(parallel, ranks);
    PreparedContext out{
        .shape     = ranks.front().shape,
        .batches   = std::make_shared<const BatchExecutionPlan>(ranks.front().batches),
        .states    = {ranks.front().state},
        .directory = std::make_unique<context::ContextDirectory>(spec),
        .spec      = std::move(spec),
        .caches    = ranks.front().caches,
    };
    return out;
}

} // namespace ninfer::models::inkling2
