#pragma once

#include "runtime/contract/resources.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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

[[nodiscard]] std::size_t scoring_context_capacity_limit(std::size_t minimum, std::size_t reference,
                                                         std::size_t block_bytes);

// Tighten a fixed-device-plus-backing ceiling, preserving a feasible minimum.
[[nodiscard]] std::size_t reduced_runtime_limit(const ContextCapacityResolution&,
                                                std::size_t backing_reduction);

// Available bytes are net allowances supplied by startup budgeting. Capacity selection charges
// model resources and their metadata; it does not reserve startup headroom again.
[[nodiscard]] ContextCapacityResolution
resolve_context_capacity(const ContextCapacityPolicy&, const ContextCapacityPlan&,
                         std::size_t available_runtime_bytes, std::size_t available_host_bytes,
                         bool shared_physical_memory, bool owns_control_memory = true,
                         EnginePurpose purpose = EnginePurpose::Generation);

struct RankCapacityOffer {
    ContextCapacityPlan plan;
    std::size_t available_bytes        = 0;
    std::size_t available_host_bytes   = SIZE_MAX; // Raw node RAM before payload reservations.
    bool shared_physical_memory        = false;
    std::uint32_t node                 = 0;
    int device                         = 0;
    std::size_t graph_reserve_bytes    = 0;
    std::size_t maximum_runtime_bytes  = SIZE_MAX; // Fixed device bytes + Context backing.
    bool physical_placement_set        = false;
    std::size_t host_reservation_bytes = 0;
};

struct CoordinatorMemoryOffer {
    std::uint32_t node                  = 0;
    std::size_t available_host_bytes    = SIZE_MAX;
    std::size_t available_unified_bytes = SIZE_MAX; // After planning headroom, before reservations.
    bool shared_physical_memory         = false;
    std::size_t frontend_host_bytes     = 0;
    std::size_t maximum_directory_bytes = SIZE_MAX;
};

struct ClusterHostReservations {
    std::map<std::uint32_t, std::size_t> by_node;
    std::size_t coordinator_directory_bytes = 0;
};

// With no solved capacities, returns the mandatory reservations independent of capacity.
[[nodiscard]] ClusterHostReservations
cluster_host_reservations(std::span<const RankCapacityOffer>, const CoordinatorMemoryOffer&,
                          std::span<const ContextCapacityResolution> = {});

// Solves Context capacities across all cluster ranks at a single post-weight/post-comm physical
// sampling instant: co-located ranks sharing a discrete GPU or UMA node share one physical pool
// via block-aligned water-filling, and coordinator_node pays for all replicas' ContextDirectory
// metadata budgets (even when a replica or the entire worker set is remote).
[[nodiscard]] std::vector<ContextCapacityResolution>
resolve_cluster_context_capacity(const ContextCapacityPolicy&, std::span<const RankCapacityOffer>,
                                 const CoordinatorMemoryOffer&,
                                 EnginePurpose purpose = EnginePurpose::Generation);

// Single-group helper retained for unit/single-replica callers; delegates to cluster solver.
[[nodiscard]] std::vector<ContextCapacityResolution>
resolve_group_context_capacity(const ContextCapacityPolicy&, std::span<const RankCapacityOffer>,
                               std::optional<std::size_t> control_memory_owner);

[[nodiscard]] std::uint64_t
estimate_context_token_capacity(std::span<const ContextCapacityResolution> capacities);

} // namespace ninfer::runtime
