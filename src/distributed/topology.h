#pragma once

#include "ninfer/deployment.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::distributed {

using RankId  = std::uint32_t;
using GroupId = std::uint32_t;

enum class GroupUsage : std::uint8_t { General, PeerExchange };

struct GroupDescription {
    GroupId id = 0;
    std::vector<RankId> members;
    GroupUsage usage = GroupUsage::General;

    [[nodiscard]] std::uint32_t index(RankId rank) const;
};

using PhysicalPoolKey = std::pair<std::uint32_t, int>;

struct RankIdentity {
    RankId rank                = 0;
    std::uint32_t node         = 0, local_rank = 0;
    int device                 = 0;
    ParallelCoordinate tp, ep, dp, pp, adp;
    std::uint64_t shard_digest = 0;

    [[nodiscard]] PhysicalPoolKey memory_pool(bool shared_physical_memory) const noexcept {
        return {node, shared_physical_memory ? -1 : device};
    }
};

struct ReplicaTopology {
    std::uint32_t replica_id        = 0;
    std::uint32_t failure_domain_id = 0;
    RankId result_owner             = 0;
    std::vector<RankId> ranks;
    std::vector<std::vector<RankId>> stage_ranks;

    [[nodiscard]] std::uint32_t local_index(RankId global_rank) const;
    [[nodiscard]] RankId global_rank(std::uint32_t local_index) const;
    [[nodiscard]] bool contains(RankId global_rank) const noexcept;
};

struct RoleGroup {
    std::string role;
    GroupId group = 0;
};

// Ordered physical groups are distinct from model roles. Several roles may use one group.
struct ExecutionTopology {
    std::uint64_t plan = 0;
    std::vector<RankIdentity> ranks;
    std::vector<ReplicaTopology> replicas;
    std::vector<GroupDescription> groups;
    std::vector<RoleGroup> roles;

    void validate(const DeploymentOptions&) const;
    [[nodiscard]] const RankIdentity& rank(RankId) const;
    [[nodiscard]] const ReplicaTopology& replica(std::uint32_t replica_id) const;
    [[nodiscard]] const ReplicaTopology& replica_for_rank(RankId global_rank) const;
    [[nodiscard]] const GroupDescription& group(GroupId) const;
    [[nodiscard]] const GroupDescription& role_group(std::string_view role, RankId member) const;
    // Adjacent stages communicate between corresponding tensor lanes. Members are ordered
    // producer first, consumer second.
    [[nodiscard]] const GroupDescription* pipeline_input(RankId rank) const;
    [[nodiscard]] const GroupDescription* pipeline_output(RankId rank) const;
    void add_role(std::string_view role, std::span<const RankId> members,
                  GroupUsage usage = GroupUsage::General);
};

[[nodiscard]] ExecutionTopology build_cluster_topology(const DeploymentOptions&);
[[nodiscard]] std::uint64_t compute_topology_plan_digest(std::span<const std::byte> artifact_id,
                                                         const ExecutionTopology&);
[[nodiscard]] std::string process_name(const RankIdentity&);
[[nodiscard]] std::string rank_description(const RankIdentity&);

} // namespace ninfer::distributed
