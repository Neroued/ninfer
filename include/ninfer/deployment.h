#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace ninfer {

struct ParallelOptions {
    std::uint32_t tp = 1, ep = 1, dp = 1, pp = 1;
};

struct DeploymentNode {
    std::uint32_t id = 0;
    // Empty for the coordinator node; otherwise an OpenSSH destination.
    std::string ssh;
    std::filesystem::path worker;
    std::filesystem::path artifact;
    std::map<std::string, std::string, std::less<>> environment;
};

struct DeploymentRank {
    std::uint32_t rank = 0, node = 0;
    int device = 0;
};

struct DeploymentOptions {
    std::uint32_t coordinator_node = 0;
    std::string control_host;
    std::uint16_t control_port        = 0;
    std::uint32_t connect_timeout_ms  = 60'000;
    std::uint32_t shutdown_timeout_ms = 30'000;
    ParallelOptions parallel;
    std::vector<DeploymentNode> nodes;
    std::vector<DeploymentRank> ranks;
};

// Validates deployment identities, paths and references. Model parallel geometry is resolved
// by the model planner, independently of this process/device mapping.
void validate_deployment(const DeploymentOptions&);

} // namespace ninfer
