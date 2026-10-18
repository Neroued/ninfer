#pragma once

#include "ninfer/deployment.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ninfer::distributed {

using RankId  = std::uint32_t;
using GroupId = std::uint32_t;

struct GroupDescription {
    GroupId id = 0;
    std::vector<RankId> members;

    [[nodiscard]] std::uint32_t index(RankId rank) const;
};

struct ParallelCoordinate {
    std::uint32_t rank = 0, size = 1;
};

struct RankIdentity {
    RankId rank        = 0;
    std::uint32_t node = 0, local_rank = 0;
    int device = 0;
    ParallelCoordinate tp, ep, dp, pp;
};

struct RoleGroup {
    std::string role;
    GroupId group = 0;
};

// Ordered physical groups are distinct from model roles. Several roles may use one group.
struct ExecutionTopology {
    std::uint64_t plan = 0;
    RankId output_rank = 0;
    std::vector<RankIdentity> ranks;
    std::vector<GroupDescription> groups;
    std::vector<RoleGroup> roles;

    void validate(const DeploymentOptions&) const;
    [[nodiscard]] const RankIdentity& rank(RankId) const;
    [[nodiscard]] const GroupDescription& group(GroupId) const;
};

[[nodiscard]] std::string process_name(const RankIdentity&);
[[nodiscard]] std::string rank_description(const RankIdentity&);

} // namespace ninfer::distributed
