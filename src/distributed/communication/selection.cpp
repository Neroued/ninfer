#include "distributed/communication/selection.h"
#include "core/platform.h"

#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::distributed::detail {
namespace {

std::array<CommunicationBackend, 2> priority(CommunicationBackend preferred) {
    return {preferred, preferred == CommunicationBackend::Native ? CommunicationBackend::Nccl
                                                                 : CommunicationBackend::Native};
}

CommunicationBackend preference(const GroupDescription& group, CommunicationPolicy policy) {
    if (policy == CommunicationPolicy::Native) return CommunicationBackend::Native;
    if (policy == CommunicationPolicy::Nccl) return CommunicationBackend::Nccl;
    // Two-rank Spark measurements favor native across the supported payload range.
    if (NINFER_TARGET_SPARK && group.members.size() == 2) return CommunicationBackend::Native;
    return CommunicationBackend::Nccl;
}

std::vector<CollectiveRoute> collective_preferences(const CollectiveRequirement& counts,
                                                    const GroupDescription& group,
                                                    CommunicationPolicy policy) {
    return {{counts, preference(group, policy)}};
}

Backend* candidate(std::span<Backend* const> candidates, CommunicationBackend kind) {
    const auto found = std::find_if(candidates.begin(), candidates.end(),
                                    [&](const auto* backend) { return backend->kind() == kind; });
    return found == candidates.end() ? nullptr : *found;
}

} // namespace

std::vector<CollectiveRoute> select_collectives(std::span<const CollectiveRoute> preferences,
                                                std::span<Backend* const> candidates) {
    std::vector<CollectiveRoute> result;
    for (const auto& preferred : preferences) {
        auto remaining = preferred.counts;
        bool covered   = false;
        for (const auto kind : priority(preferred.backend)) {
            const auto* backend = candidate(candidates, kind);
            const auto supports = [&](std::size_t count) {
                return backend->supports({remaining.operation, remaining.dtype, count,
                                          remaining.reduction, remaining.root});
            };
            if (!backend || !supports(remaining.min_count)) continue;
            auto lo = std::size_t{0},
                 hi = (remaining.max_count - remaining.min_count) / remaining.step;
            while (lo < hi) {
                const auto mid = lo + (hi - lo + 1) / 2;
                if (supports(remaining.min_count + mid * remaining.step))
                    lo = mid;
                else
                    hi = mid - 1;
            }
            auto range      = remaining;
            range.max_count = remaining.min_count + lo * remaining.step;
            result.push_back({range, kind});
            if (range.max_count == remaining.max_count) {
                covered = true;
                break;
            }
            remaining.min_count = range.max_count + remaining.step;
        }
        if (!covered)
            throw std::invalid_argument(
                "prepared backends do not cover the collective count range");
    }
    return result;
}

GroupRouting select_routing(const GroupRequirements& requirements, const GroupDescription& group,
                            CommunicationPolicy policy, std::span<Backend* const> candidates,
                            bool peer_work) {
    GroupRouting result;
    if (group.members.size() == 1) return result;
    for (const auto& counts : requirements.collectives) {
        auto routes = select_collectives(collective_preferences(counts, group, policy), candidates);
        result.collectives.insert(result.collectives.end(), routes.begin(), routes.end());
    }
    if (peer_work) {
        for (const auto kind : priority(preference(group, policy))) {
            const auto* backend = candidate(candidates, kind);
            if (backend && backend->supports_peers(requirements.peers)) {
                result.peers = kind;
                break;
            }
        }
        if (!result.peers)
            throw std::invalid_argument("prepared backends do not cover peer transfers");
    }
    return result;
}

} // namespace ninfer::distributed::detail
