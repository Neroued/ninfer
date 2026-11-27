#include "distributed/topology.h"

#include <algorithm>
#include <bit>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace ninfer::distributed {
namespace {

class Fnv1a64 {
public:
    void u8(std::uint8_t v) noexcept {
        state_ ^= v;
        state_ *= 1099511628211ULL;
    }

    void u64(std::uint64_t v) noexcept {
        for (unsigned i = 0; i < 8; ++i) u8(static_cast<std::uint8_t>((v >> (i * 8)) & 0xffU));
    }

    void bytes(std::span<const std::byte> data) noexcept {
        u64(data.size());
        for (const auto b : data) u8(std::to_integer<std::uint8_t>(b));
    }

    void string(std::string_view s) noexcept {
        bytes(std::as_bytes(std::span(s.data(), s.size())));
    }

    [[nodiscard]] std::uint64_t finish() const noexcept { return state_ ? state_ : 1ULL; }

private:
    std::uint64_t state_ = 14695981039346656037ULL;
};

bool same_coordinate(ParallelCoordinate a, ParallelCoordinate b) noexcept {
    return a.rank == b.rank && a.size == b.size;
}

} // namespace

std::uint32_t GroupDescription::index(RankId rank) const {
    const auto found = std::find(members.begin(), members.end(), rank);
    if (found == members.end()) throw std::invalid_argument("rank is not a group member");
    return static_cast<std::uint32_t>(found - members.begin());
}

std::uint32_t ReplicaTopology::local_index(RankId global_rank) const {
    const auto found = std::find(ranks.begin(), ranks.end(), global_rank);
    if (found == ranks.end()) throw std::invalid_argument("rank does not belong to replica");
    return static_cast<std::uint32_t>(found - ranks.begin());
}

RankId ReplicaTopology::global_rank(std::uint32_t index) const {
    if (index >= ranks.size()) throw std::invalid_argument("replica local rank out of range");
    return ranks[index];
}

bool ReplicaTopology::contains(RankId global_rank) const noexcept {
    return std::find(ranks.begin(), ranks.end(), global_rank) != ranks.end();
}

const RankIdentity& ExecutionTopology::rank(RankId id) const {
    if (id >= ranks.size() || ranks[id].rank != id)
        throw std::invalid_argument("unknown topology rank");
    return ranks[id];
}

const ReplicaTopology& ExecutionTopology::replica(std::uint32_t replica_id) const {
    if (replica_id >= replicas.size() || replicas[replica_id].replica_id != replica_id)
        throw std::invalid_argument("unknown topology replica");
    return replicas[replica_id];
}

const ReplicaTopology& ExecutionTopology::replica_for_rank(RankId global_rank) const {
    return replica(rank(global_rank).dp.rank);
}

const GroupDescription& ExecutionTopology::group(GroupId id) const {
    const auto found = std::find_if(groups.begin(), groups.end(),
                                    [id](const auto& value) { return value.id == id; });
    if (found == groups.end()) throw std::invalid_argument("unknown communication group");
    return *found;
}

const GroupDescription& ExecutionTopology::role_group(std::string_view name, RankId member) const {
    for (const auto& binding : roles) {
        if (binding.role != name) continue;
        const auto& candidate = group(binding.group);
        if (std::ranges::find(candidate.members, member) != candidate.members.end())
            return candidate;
    }
    throw std::invalid_argument("missing communication role " + std::string(name) + " for rank " +
                                std::to_string(member));
}

void ExecutionTopology::add_role(std::string_view name, std::span<const RankId> members,
                                 GroupUsage usage) {
    if (members.size() == 1) return; // Local roles require no communicator.
    const auto same = std::ranges::find_if(groups, [&](const GroupDescription& g) {
        return g.usage == usage && std::ranges::equal(g.members, members);
    });
    if (same == groups.end()) {
        auto id = static_cast<GroupId>(groups.size());
        while (std::ranges::find(groups, id, &GroupDescription::id) != groups.end()) ++id;
        groups.push_back({id, {members.begin(), members.end()}, usage});
        roles.push_back({std::string(name), id});
    } else {
        roles.push_back({std::string(name), same->id});
    }
}

ExecutionTopology build_cluster_topology(const DeploymentOptions& raw_deployment) {
    const auto deployment = normalize_deployment(raw_deployment);
    const auto& p         = deployment.parallel;
    const auto replica_span = p.pp * p.tp;

    ExecutionTopology out;
    out.plan = 1;
    out.ranks.resize(deployment.ranks.size());
    out.replicas.resize(p.dp);

    for (std::uint32_t r = 0; r < p.dp; ++r) {
        auto& rep             = out.replicas[r];
        rep.replica_id        = r;
        rep.failure_domain_id = p.ep <= p.tp ? r : (r / (p.ep / p.tp));
        rep.stage_ranks.resize(p.pp);
        rep.ranks.reserve(replica_span);
        for (std::uint32_t s = 0; s < p.pp; ++s) {
            rep.stage_ranks[s].resize(p.tp);
            for (std::uint32_t t = 0; t < p.tp; ++t) {
                const RankId global_id = r * replica_span + s * p.tp + t;
                rep.stage_ranks[s][t]  = global_id;
                rep.ranks.push_back(global_id);

                const auto& deployed = deployment.ranks[global_id];
                const auto local = std::count_if(
                    deployment.ranks.begin(), deployment.ranks.begin() + global_id,
                    [&](const auto& other) { return other.node == deployed.node; });
                const std::uint32_t adp_rank = t / (p.tp / p.adp);
                const std::uint32_t ep_rank =
                    p.ep <= p.tp ? (t / (p.tp / p.ep))
                                 : ((r % (p.ep / p.tp)) * p.tp + t);
                out.ranks[global_id] = {global_id,
                                        deployed.node,
                                        static_cast<std::uint32_t>(local),
                                        deployed.device,
                                        {t, p.tp},
                                        {ep_rank, p.ep},
                                        {r, p.dp},
                                        {s, p.pp},
                                        {adp_rank, p.adp},
                                        0};
            }
        }
        rep.result_owner = rep.stage_ranks[p.pp - 1][0];
    }
    return out;
}

std::uint64_t compute_topology_plan_digest(std::span<const std::byte> artifact_id,
                                           const ExecutionTopology& topology) {
    Fnv1a64 hash;
    hash.bytes(artifact_id);
    hash.u64(topology.ranks.size());
    for (const auto& r : topology.ranks) {
        hash.u64(r.rank);
        hash.u64(r.node);
        hash.u64(r.local_rank);
        hash.u64(std::bit_cast<std::uint32_t>(static_cast<std::int32_t>(r.device)));
        for (const auto c : {r.tp, r.ep, r.dp, r.pp, r.adp}) {
            hash.u64(c.rank);
            hash.u64(c.size);
        }
        hash.u64(r.shard_digest);
    }
    hash.u64(topology.replicas.size());
    for (const auto& rep : topology.replicas) {
        hash.u64(rep.replica_id);
        hash.u64(rep.failure_domain_id);
        hash.u64(rep.result_owner);
        hash.u64(rep.ranks.size());
        for (const auto id : rep.ranks) hash.u64(id);
        hash.u64(rep.stage_ranks.size());
        for (const auto& stage : rep.stage_ranks) {
            hash.u64(stage.size());
            for (const auto id : stage) hash.u64(id);
        }
    }
    hash.u64(topology.groups.size());
    for (const auto& g : topology.groups) {
        hash.u64(g.id);
        hash.u8(static_cast<std::uint8_t>(g.usage));
        hash.u64(g.members.size());
        for (const auto m : g.members) hash.u64(m);
    }
    hash.u64(topology.roles.size());
    for (const auto& role : topology.roles) {
        hash.string(role.role);
        hash.u64(role.group);
    }
    return hash.finish();
}

void ExecutionTopology::validate(const DeploymentOptions& deployment) const {
    const auto expected = build_cluster_topology(deployment);
    if (!plan || ranks.size() != expected.ranks.size() ||
        replicas.size() != expected.replicas.size())
        throw std::invalid_argument("invalid execution topology identity");

    for (std::size_t i = 0; i < expected.ranks.size(); ++i) {
        const auto& exp = expected.ranks[i];
        const auto& act = rank(exp.rank);
        if (act.node != exp.node || act.device != exp.device)
            throw std::invalid_argument("topology differs from deployed rank placement");
        if (act.local_rank != exp.local_rank)
            throw std::invalid_argument("topology has an invalid local rank");
        for (const auto coordinate : {act.tp, act.ep, act.dp, act.pp, act.adp})
            if (!coordinate.size || coordinate.rank >= coordinate.size)
                throw std::invalid_argument("invalid parallel coordinate");
        if (!same_coordinate(act.tp, exp.tp) || !same_coordinate(act.ep, exp.ep) ||
            !same_coordinate(act.dp, exp.dp) || !same_coordinate(act.pp, exp.pp) ||
            !same_coordinate(act.adp, exp.adp))
            throw std::invalid_argument("topology differs from requested parallel geometry");
    }

    for (std::size_t r = 0; r < expected.replicas.size(); ++r) {
        const auto& exp = expected.replicas[r];
        const auto& act = replica(exp.replica_id);
        if (act.failure_domain_id != exp.failure_domain_id || act.result_owner != exp.result_owner ||
            act.ranks != exp.ranks || act.stage_ranks != exp.stage_ranks)
            throw std::invalid_argument("topology replica layout differs from cluster geometry");
    }

    std::set<GroupId> ids;
    for (const auto& g : groups) {
        if (g.usage != GroupUsage::General && g.usage != GroupUsage::PeerExchange)
            throw std::invalid_argument("invalid communication group usage");
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
        for (auto r : group(role.group).members)
            if (!roles_by_rank[role.role].insert(r).second)
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
