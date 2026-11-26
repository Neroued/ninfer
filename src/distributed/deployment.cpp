#include "ninfer/deployment.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <utility>

namespace ninfer {
namespace {

std::optional<std::uint64_t> checked_product(std::uint64_t a, std::uint64_t b) noexcept {
    if (b && a > UINT64_MAX / b) return std::nullopt;
    return a * b;
}

} // namespace

void validate_deployment(const DeploymentOptions& options) {
    const auto fail = [](const char* message) {
        throw std::invalid_argument(std::string("distributed: ") + message);
    };
    if (options.nodes.empty() || options.ranks.empty()) fail("nodes and ranks must not be empty");
    if (options.control_host.empty()) fail("control_host must not be empty");
    if (!options.connect_timeout_ms || !options.shutdown_timeout_ms)
        fail("connection and shutdown timeouts must be positive");
    const auto& p = options.parallel;
    if (!p.tp || !p.ep || !p.dp || !p.pp || !p.adp) fail("parallel sizes must be positive");
    const auto dp_pp = checked_product(p.dp, p.pp);
    const auto world = dp_pp ? checked_product(*dp_pp, p.tp) : std::nullopt;
    if (!world || *world != options.ranks.size())
        fail("dp * pp * tp must match deployed rank count");
    if (p.tp % p.adp != 0) fail("tp must be divisible by adp");
    const auto dp_tp     = checked_product(p.dp, p.tp);
    const bool pp1_valid = p.pp == 1 && (p.tp % p.ep == 0 ||
                                         (p.ep % p.tp == 0 && dp_tp && (*dp_tp % p.ep == 0)));
    const bool ppn_valid = p.dp == 1 && p.pp > 1 && p.tp % p.ep == 0;
    if (!pp1_valid && !ppn_valid) fail("unsupported parallel nesting combination");

    std::set<std::uint32_t> nodes;
    for (const auto& node : options.nodes) {
        if (!nodes.insert(node.id).second) fail("duplicate node ID");
        if (!node.worker.is_absolute() || !node.artifact.is_absolute())
            fail("node worker and artifact paths must be absolute");
        if (node.id == options.coordinator_node) {
            if (!node.ssh.empty()) fail("coordinator node must use local launch");
        } else if (node.ssh.empty() || node.ssh.front() == '-') {
            fail("remote node requires an SSH destination");
        }
        for (const auto& [key, value] : node.environment) {
            if (key.empty() || key.find('=') != std::string::npos ||
                key.find('\0') != std::string::npos || value.find('\0') != std::string::npos)
                fail("invalid worker environment assignment");
        }
    }
    if (!nodes.contains(options.coordinator_node)) fail("coordinator node is absent");
    std::set<std::uint32_t> ranks;
    std::set<std::pair<std::uint32_t, int>> devices;
    for (const auto& rank : options.ranks) {
        if (rank.rank >= options.ranks.size() || !ranks.insert(rank.rank).second)
            fail("global ranks must be unique and cover [0,world_size)");
        if (!nodes.contains(rank.node)) fail("rank refers to an absent node");
        if (rank.device < 0) fail("rank device must be nonnegative");
        if (!options.allow_shared_device && !devices.emplace(rank.node, rank.device).second)
            fail("each rank requires a distinct nonnegative node/device pair");
    }
    for (const auto node : nodes)
        if (std::none_of(options.ranks.begin(), options.ranks.end(),
                         [node](const auto& rank) { return rank.node == node; }))
            fail("node has no ranks");
}

DeploymentOptions normalize_deployment(DeploymentOptions options) {
    validate_deployment(options);
    std::sort(options.ranks.begin(), options.ranks.end(),
              [](const auto& a, const auto& b) { return a.rank < b.rank; });
    return options;
}

} // namespace ninfer
