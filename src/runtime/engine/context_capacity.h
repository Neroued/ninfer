#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

class ContextBudgetError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

[[nodiscard]] constexpr bool keeps_context_capacity(std::size_t narrow, std::size_t baseline) {
    return narrow >= baseline - baseline / 20;
}

// Available bytes are net allowances supplied by startup budgeting. Capacity selection charges
// model resources and their metadata; it does not reserve startup headroom again.
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
