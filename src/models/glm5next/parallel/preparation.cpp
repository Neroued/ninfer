#include "models/glm5next/parallel/preparation.h"

#include <algorithm>

namespace ninfer::models::glm5next {

context::ContextDirectorySpec
plan_replica_context(std::span<const RankPlan> expected_ranks,
                     std::span<const RankResourceDescription> ranks) {
    if (ranks.empty() || ranks.size() != expected_ranks.size())
        throw std::invalid_argument("GLM5Next prepared rank count differs from its replica plan");
    const auto& geometry = expected_ranks.front().placement;
    const AttentionOwners groups(geometry.tp.size, geometry.adp.size, geometry.pp.size);
    if (std::uint64_t(geometry.tp.size) * geometry.pp.size != ranks.size())
        throw std::invalid_argument("GLM5Next rank count differs from its pipeline stages");
    context::ContextDirectorySpec out;
    out.state.logical_capacity = UINT32_MAX;
    out.state.layouts.resize(groups.size());
    for (std::size_t owner = 0; owner < groups.size(); ++owner)
        out.state.layouts[owner].variant = static_cast<std::uint32_t>(owner);
    out.pools.resize(ranks.front().context.pools.size());
    for (std::size_t rank = 0; rank < ranks.size(); ++rank) {
        const auto& r = ranks[rank];
        const auto& c = r.context;
        if (r.observation.rank != expected_ranks[rank].placement.global_rank || r.rank != rank ||
            expected_ranks[rank].placement.tp.rank != rank % geometry.tp.size ||
            expected_ranks[rank].placement.pp.rank != groups.stage(rank) ||
            r.shape != ranks.front().shape || r.batches != ranks.front().batches ||
            c.ranks.size() != 1 || c.ranks.front().rank != rank ||
            expected_ranks[rank].placement.adp.rank != groups.owner(rank) ||
            expected_ranks[rank].placement.adp.size != groups.size() ||
            r.batches.attention_owners != groups.size() ||
            c.state.layouts.size() != groups.size() || c.pools.size() != out.pools.size() ||
            r.caches != ranks[groups.stage(rank) * geometry.tp.size].caches ||
            r.caches.size() != out.pools.size())
            throw std::invalid_argument("GLM5Next rank resources disagree with the group layout");
        out.ranks.push_back(c.ranks.front());
        out.metadata_budget_bytes += c.metadata_budget_bytes;
        out.state.logical_capacity = std::min(out.state.logical_capacity, c.state.logical_capacity);
        for (std::size_t owner = 0; owner < groups.size(); ++owner) {
            const auto& layout = c.state.layouts[owner];
            if (layout.variant != owner || layout.parts.size() != 1 || layout.parts[0].rank != rank)
                throw std::invalid_argument("GLM5Next State variant has inconsistent rank parts");
            const auto& expected = owner == groups.owner(rank) ? r.state.owner : r.state.common;
            if (layout.parts[0].image_bytes != expected.bytes ||
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
                source.page_capacity != target.page_capacity ||
                source.parts.size() != std::size_t(!r.caches[pool].planes.empty()))
                throw std::invalid_argument("GLM5Next history layouts differ between ranks");
            if (!source.parts.empty()) {
                const auto& first =
                    ranks[groups.stage(rank) * geometry.tp.size].context.pools[pool].parts;
                if (source.parts[0].rank != rank || first.empty() ||
                    first[0].record != source.parts[0].record)
                    throw std::invalid_argument(
                        "GLM5Next stage disagrees on its local history part");
                target.parts.push_back(source.parts[0]);
            }
        }
    }
    for (auto& pool : out.pools) {
        for (std::uint32_t owner = 0; owner < groups.size(); ++owner) {
            std::uint32_t pages = UINT32_MAX;
            for (const auto rank : groups.members(owner))
                if (const auto& local = ranks[rank].context.pools[pool.pool_id];
                    !local.parts.empty())
                    pages = std::min(pages, local.logical_pages);
            if (pages == UINT32_MAX)
                throw std::invalid_argument("GLM5Next history has no physical page owner");
            if (pages > UINT32_MAX - pool.logical_pages)
                throw std::overflow_error("GLM5Next logical page capacity exceeds uint32");
            pool.logical_pages += pages;
        }
        pool.histories =
            context::HistoryDirectory::capacity_for_metadata(out.metadata_budget_bytes);
    }
    return out;
}

PreparedContext prepare_replica_context(std::span<const RankPlan> expected_ranks,
                                        std::span<const RankResourceDescription> ranks) {
    auto spec = plan_replica_context(expected_ranks, ranks);
    PreparedContext out{ranks.front().shape,
                        std::make_shared<const BatchExecutionPlan>(ranks.front().batches),
                        {},
                        ranks.front().vision_max_features,
                        std::make_unique<context::ContextDirectory>(spec),
                        std::move(spec),
                        {}};
    out.caches.resize(ranks.front().caches.size());
    for (std::size_t r = 0; r < ranks.size(); r += expected_ranks.front().placement.tp.size)
        for (std::size_t pool = 0; pool < out.caches.size(); ++pool)
            out.caches[pool].push_back(ranks[r].caches[pool]);
    for (const auto& rank : ranks) {
        out.states.push_back(rank.state);
        if (rank.rank < expected_ranks.front().placement.tp.size)
            out.vision_max_features = std::min(out.vision_max_features, rank.vision_max_features);
    }
    return out;
}

} // namespace ninfer::models::glm5next
