#pragma once

#include "models/common/parallel.h"

#include <algorithm>
#include <memory>

namespace ninfer::models::moe {

// Peer order is the communication group's order. Each peer lists its local logical slots.
struct ExpertDomain {
    std::uint32_t experts = 0, intermediate = 0;
    std::vector<distributed::RankId> members;
    std::vector<std::vector<ExpertPartition>> banks;

    [[nodiscard]] std::uint64_t bank_signature(std::size_t bank) const;
    void validate() const;
    void validate_local(distributed::RankId, std::span<const ExpertPartition>) const;
};

template <class RankPlan>
void bind_expert_domains(const distributed::ExecutionTopology& topology,
                         std::vector<RankPlan>& ranks, std::uint32_t experts,
                         std::uint32_t intermediate) {
    for (const auto& role : topology.roles) {
        if (role.role != "expert") continue;
        const auto& group    = topology.group(role.group);
        if (ranks.at(group.members.front()).experts.empty()) continue;
        auto domain          = std::make_shared<ExpertDomain>();
        domain->members      = group.members;
        domain->experts      = experts;
        domain->intermediate = intermediate;
        domain->banks.resize(ranks.at(group.members.front()).experts.size());
        for (const auto member : group.members) {
            const auto& local = ranks.at(member).experts;
            if (local.size() != domain->banks.size())
                throw std::invalid_argument("expert dispatch bank counts differ");
            for (std::size_t b = 0; b < local.size(); ++b) domain->banks[b].push_back(local[b]);
        }
        domain->validate();
        for (const auto member : group.members) ranks.at(member).expert_domain = domain;
    }
}

} // namespace ninfer::models::moe
