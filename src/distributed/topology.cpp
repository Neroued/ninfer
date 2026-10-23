#include "distributed/topology.h"

#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>

namespace ninfer::distributed {

std::uint32_t GroupDescription::index(RankId rank) const {
    const auto found = std::find(members.begin(), members.end(), rank);
    if (found == members.end()) throw std::invalid_argument("rank is not a group member");
    return static_cast<std::uint32_t>(found - members.begin());
}

const RankIdentity& ExecutionTopology::rank(RankId id) const {
    if (id >= ranks.size() || ranks[id].rank != id)
        throw std::invalid_argument("unknown topology rank");
    return ranks[id];
}

const GroupDescription& ExecutionTopology::group(GroupId id) const {
    const auto found = std::find_if(groups.begin(), groups.end(),
                                    [id](const auto& value) { return value.id == id; });
    if (found == groups.end()) throw std::invalid_argument("unknown communication group");
    return *found;
}

void ExecutionTopology::validate(const DeploymentOptions& deployment) const {
    validate_deployment(deployment);
    if (!plan || ranks.size() != deployment.ranks.size() || output_rank >= ranks.size())
        throw std::invalid_argument("invalid execution topology identity");
    for (const auto& expected : deployment.ranks) {
        const auto& actual = rank(expected.rank);
        if (actual.node != expected.node || actual.device != expected.device)
            throw std::invalid_argument("topology differs from deployed rank placement");
        const auto local =
            std::count_if(deployment.ranks.begin(), deployment.ranks.end(), [&](const auto& r) {
                return r.node == actual.node && r.rank < actual.rank;
            });
        if (actual.local_rank != static_cast<std::uint32_t>(local))
            throw std::invalid_argument("topology has an invalid local rank");
        for (const auto coordinate : {actual.tp, actual.ep, actual.dp, actual.pp, actual.adp})
            if (!coordinate.size || coordinate.rank >= coordinate.size)
                throw std::invalid_argument("invalid parallel coordinate");
        const auto& p = deployment.parallel;
        if (actual.tp.size != p.tp || actual.ep.size != p.ep || actual.dp.size != p.dp ||
            actual.pp.size != p.pp || actual.adp.size != p.attention_dp)
            throw std::invalid_argument("topology differs from requested parallel sizes");
    }
    std::set<GroupId> ids;
    for (const auto& g : groups) {
        if (g.members.empty() || !ids.insert(g.id).second)
            throw std::invalid_argument("empty or duplicate communication group");
        std::set<RankId> members;
        for (auto r : g.members)
            if (r >= ranks.size() || !members.insert(r).second)
                throw std::invalid_argument("invalid communication group membership");
    }
    std::map<std::string, std::set<RankId>> roles_by_rank;
    for (const auto& role : roles) {
        if (role.role.empty()) throw std::invalid_argument("empty communication role");
        for (auto rank : group(role.group).members)
            if (!roles_by_rank[role.role].insert(rank).second)
                throw std::invalid_argument("rank has multiple groups for one role");
    }
}

std::string process_name(const RankIdentity& r) {
    return "ninfer:worker:dp" + std::to_string(r.dp.rank) + ":tp" + std::to_string(r.tp.rank) +
           ":ep" + std::to_string(r.ep.rank) + ":pp" + std::to_string(r.pp.rank) + ":adp" +
           std::to_string(r.adp.rank);
}

std::string rank_description(const RankIdentity& r) {
    const auto coordinate = [](ParallelCoordinate c) {
        return std::to_string(c.rank) + "/" + std::to_string(c.size);
    };
    return "node=" + std::to_string(r.node) + " rank=" + std::to_string(r.rank) +
           " local=" + std::to_string(r.local_rank) + " cuda=" + std::to_string(r.device) +
           " | TP=" + coordinate(r.tp) + " EP=" + coordinate(r.ep) + " DP=" + coordinate(r.dp) +
           " PP=" + coordinate(r.pp) + " ADP=" + coordinate(r.adp);
}

} // namespace ninfer::distributed
