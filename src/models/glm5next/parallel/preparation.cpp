#include "models/glm5next/parallel/preparation.h"

#include <algorithm>

namespace ninfer::models::glm5next {

context::ContextDirectorySpec plan_group_context(const ParallelPlan& parallel,
                                                 std::span<const RankResourceDescription> ranks) {
    if (ranks.empty() || ranks.size() != parallel.ranks.size())
        throw std::invalid_argument("GLM5Next prepared rank count differs from its parallel plan");
    context::ContextDirectorySpec out;
    out.state.logical_capacity = UINT32_MAX;
    out.state.layouts.resize(ranks.size());
    for (std::size_t owner = 0; owner < ranks.size(); ++owner)
        out.state.layouts[owner].variant = static_cast<std::uint32_t>(owner);
    out.pools.resize(ranks.front().context.pools.size());
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        const auto& r = ranks[rank];
        const auto& c = r.context;
        if (r.rank != parallel.ranks[rank].rank || r.rank != rank ||
            r.shape != ranks.front().shape || c.ranks.size() != 1 || c.ranks.front().rank != rank ||
            c.state.layouts.size() != ranks.size() || c.pools.size() != out.pools.size())
            throw std::invalid_argument("GLM5Next rank resources disagree with the group layout");
        out.ranks.push_back(c.ranks.front());
        out.metadata_budget_bytes += c.metadata_budget_bytes;
        out.state.logical_capacity = std::min(out.state.logical_capacity, c.state.logical_capacity);
        for (std::size_t owner = 0; owner < ranks.size(); ++owner) {
            const auto& layout = c.state.layouts[owner];
            if (layout.variant != owner || layout.parts.size() != 1 || layout.parts[0].rank != rank)
                throw std::invalid_argument("GLM5Next State variant has inconsistent rank parts");
            const auto& expected = owner == rank ? r.state.owner : r.state.common;
            if (layout.parts[0].image_bytes != expected.image_bytes ||
                layout.parts[0].alignment != expected.alignment)
                throw std::invalid_argument("GLM5Next State part differs from its field layout");
            out.state.layouts[owner].parts.push_back(layout.parts[0]);
        }
        for (std::size_t pool = 0; pool < out.pools.size(); ++pool) {
            const auto& source = c.pools[pool];
            auto& target       = out.pools[pool];
            if (rank == 0) {
                target               = source;
                target.logical_pages = 0;
                target.parts.clear();
            }
            if (source.pool_id != pool || source.table_rows != target.table_rows ||
                source.page_capacity != target.page_capacity || source.parts.size() != 1 ||
                source.parts[0].rank != rank ||
                source.parts[0].record != ranks.front().context.pools[pool].parts[0].record)
                throw std::invalid_argument("GLM5Next history layouts differ between ranks");
            if (source.logical_pages > UINT32_MAX - target.logical_pages)
                throw std::overflow_error("GLM5Next logical page capacity exceeds uint32");
            target.logical_pages += source.logical_pages;
            target.parts.push_back(source.parts[0]);
        }
    }
    const auto histories =
        4ULL * out.state.logical_capacity + 3ULL * ranks.front().shape.max_concurrency + 2;
    if (histories > UINT32_MAX)
        throw std::overflow_error("GLM5Next history capacity exceeds uint32");
    for (auto& pool : out.pools) pool.histories = static_cast<std::uint32_t>(histories);
    return out;
}

} // namespace ninfer::models::glm5next
