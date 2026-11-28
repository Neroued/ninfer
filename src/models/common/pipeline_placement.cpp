#include "models/common/pipeline_placement.h"
#include <algorithm>
#include <map>
#include <tuple>

namespace ninfer::models {
std::optional<std::vector<IndexRange>>
fit_pipeline_layers(const distributed::ExecutionTopology& hardware, const EngineOptions& options,
                    std::span<const distributed::RankMemoryOffer> offers,
                    std::span<const IndexRange> current,
                    std::span<const PipelineLayerDemand> demands, const PipelineShardBytes& size) {
    const auto& stages = hardware.replicas.front().stage_ranks;
    if (offers.size() != hardware.ranks.size())
        throw std::invalid_argument("incomplete pipeline rank capacities");
    std::map<distributed::PhysicalPoolKey, std::size_t> pools;
    std::vector<std::size_t> allowance, ranks_per_pool, rank_pool;
    for (const auto& rank : hardware.ranks) {
        const auto& offer      = offers[rank.rank];
        const auto key         = rank.memory_pool(offer.shared_physical_memory);
        const auto [it, added] = pools.try_emplace(key, pools.size());
        const auto pool        = it->second;
        if (added) {
            allowance.push_back(offer.available_bytes);
            ranks_per_pool.push_back(0);
        }
        allowance[pool] = std::min<std::size_t>(allowance[pool], offer.available_bytes);
        ++ranks_per_pool[pool];
        rank_pool.push_back(pool);
    }
    for (std::size_t i = 0; i < allowance.size(); ++i) {
        auto& free = allowance[i];
        free -= std::min(free, options.memory_headroom_bytes);
        if (options.context_capacity.mode == ContextCapacityMode::Explicit) {
            const auto per_rank = options.context_capacity.explicit_bytes;
            free = per_rank > free / ranks_per_pool[i] ? 0 : free - per_rank * ranks_per_pool[i];
        }
    }
    // Materialization includes the embedding/head only on their actual pipeline endpoints.
    // The CPU load planner measures the encoded TP/EP shards without reading tensor payloads.
    using Interval = std::tuple<std::size_t, std::uint32_t, std::uint32_t>;
    std::map<Interval, std::vector<std::size_t>> charges;
    const auto fits = [&](std::span<const IndexRange> prefix) {
        auto remaining = allowance;
        for (std::size_t s = 0; s < prefix.size(); ++s) {
            const auto range          = prefix[s];
            const auto [it, inserted] = charges.try_emplace({s, range.begin, range.size});
            auto& bytes               = it->second;
            if (inserted) {
                bytes.resize(allowance.size());
                for (const auto r : stages[s]) {
                    const auto charge =
                        size(hardware.rank(r), range, offers[r].shared_physical_memory);
                    auto& total = bytes[rank_pool[r]];
                    if (charge > SIZE_MAX - total)
                        throw std::overflow_error("pipeline weight bytes overflow");
                    total += charge;
                }
            }
            for (std::size_t pool = 0; pool < bytes.size(); ++pool) {
                if (bytes[pool] > remaining[pool]) return false;
                remaining[pool] -= bytes[pool];
            }
        }
        return true;
    };
    if (fits(current)) return {};
    return partition_pipeline_layers(demands, options.deployment->parallel, fits);
}
} // namespace ninfer::models
