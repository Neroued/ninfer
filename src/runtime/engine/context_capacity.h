#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::runtime {

[[nodiscard]] ContextCapacityResolution
resolve_context_capacity(const ContextCapacityPolicy&, const ContextCapacityPlan&,
                         std::size_t available_runtime_bytes, std::size_t available_host_bytes,
                         bool shared_physical_memory, bool owns_control_memory = true);

struct RankCapacityOffer {
    ContextCapacityPlan plan;
    std::size_t available_bytes      = 0;
    std::size_t available_host_bytes = 0;
    bool shared_physical_memory      = false;
};

// Each rank keeps its own backing capacity. The control owner pays for all directories.
[[nodiscard]] std::vector<ContextCapacityResolution>
resolve_group_context_capacity(const ContextCapacityPolicy&, std::span<const RankCapacityOffer>,
                               std::optional<std::size_t> control_memory_owner);

[[nodiscard]] std::uint64_t
estimate_context_token_capacity(std::span<const ContextCapacityResolution> capacities);

} // namespace ninfer::runtime
