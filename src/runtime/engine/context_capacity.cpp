#include "runtime/engine/context_capacity.h"

#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::runtime {
namespace {

std::size_t checked_add(std::size_t a, std::size_t b) {
    if (b > SIZE_MAX - a) throw std::overflow_error("context reservation exceeds size_t");
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b) {
    if (b && a > SIZE_MAX / b) throw std::overflow_error("context reservation exceeds size_t");
    return a * b;
}

std::uint32_t node_for(const RankCapacityOffer& offer, std::size_t rank) {
    return offer.physical_placement_set ? offer.node : static_cast<std::uint32_t>(rank);
}

std::size_t minimum_bytes(const ContextCapacityPlan& plan, const ContextPoolGeometry& geometry) {
    if (plan.working_sets.empty())
        throw std::invalid_argument("context capacity plan has no startup working set");
    std::size_t minimum = 0;
    for (const auto& set : plan.working_sets)
        minimum = std::max(minimum, geometry.required_bytes(set.records));
    if (!minimum) throw std::invalid_argument("context startup working sets are empty");
    return minimum;
}

std::size_t control_bytes(const ContextMetadataPlan& metadata, const ContextPoolGeometry& geometry,
                          std::size_t capacity) {
    if (!metadata.bytes_per_record.empty() &&
        metadata.bytes_per_record.size() != geometry.classes.size())
        throw std::invalid_argument("context metadata classes differ from pool geometry");
    std::size_t class_bytes = 0;
    for (std::size_t i = 0; i < metadata.bytes_per_record.size(); ++i)
        class_bytes = std::max(class_bytes, checked_mul(geometry.classes[i].records_per_block,
                                                        metadata.bytes_per_record[i]));
    const auto per_block = checked_add(metadata.bytes_per_block, class_bytes);
    return checked_add(
        geometry.metadata_bytes(capacity),
        checked_add(metadata.fixed_bytes, checked_mul(capacity / geometry.block_bytes, per_block)));
}

template <class Predicate>
std::size_t largest(std::size_t low, std::size_t high, Predicate fits) {
    while (low < high) {
        const auto middle = low + (high - low) / 2 + (high - low) % 2;
        if (fits(middle))
            low = middle;
        else
            high = middle - 1;
    }
    return low;
}

void validate_policy(const ContextCapacityPolicy& policy) {
    switch (policy.mode) {
    case ContextCapacityMode::Automatic:
        if (!policy.explicit_bytes) return;
        break;
    case ContextCapacityMode::Explicit:
        if (policy.explicit_bytes) return;
        break;
    }
    throw std::invalid_argument("inconsistent Context capacity policy");
}

struct CapacityOwner {
    RankCapacityOffer offer;
    ContextPoolGeometry geometry;
    std::size_t minimum  = 0;
    std::size_t max_back = 0;
    std::size_t selected = 0;

    [[nodiscard]] std::size_t metadata(std::size_t bytes) const {
        return control_bytes(offer.plan.metadata, geometry, bytes);
    }

    [[nodiscard]] std::size_t device_consumption(std::size_t bytes) const {
        return checked_add(checked_add(offer.plan.fixed_device_bytes, offer.graph_reserve_bytes),
                           bytes);
    }
};

struct PhysicalDevicePool {
    std::uint32_t node          = 0;
    int device                  = 0; // -1 for node-wide UMA pool
    bool shared_physical_memory = false;
    bool is_coordinator_uma     = false;
    std::size_t budget_bytes    = SIZE_MAX;
    std::vector<std::size_t> ranks;
};

[[noreturn]] void insufficient(std::size_t rank, const ContextCapacityPlan& plan,
                               const ContextPoolGeometry& geometry, std::size_t available) {
    for (const auto& set : plan.working_sets) {
        const auto required = geometry.required_bytes(set.records);
        if (required > available)
            throw ContextBudgetError("Context rank " + std::to_string(rank) +
                                     " cannot hold startup working set '" + set.name +
                                     "': requires " + std::to_string(required) +
                                     " backing bytes, available " + std::to_string(available));
    }
    throw ContextBudgetError("Context rank " + std::to_string(rank) +
                             " capacity and directory metadata exceed the memory budget");
}

std::vector<ContextCapacityResolution>
solve_capacities(const ContextCapacityPolicy& policy, std::span<const RankCapacityOffer> raw_offers,
                 const CoordinatorMemoryOffer& coordinator, bool charge_control_directory,
                 EnginePurpose purpose) {
    validate_policy(policy);
    if (raw_offers.empty()) throw std::invalid_argument("Context capacity requires memory owners");

    std::vector<CapacityOwner> owners;
    owners.reserve(raw_offers.size());

    // Group ranks into shared physical device / UMA pools and per-node Host RAM pools.
    using PoolKey = std::pair<std::uint32_t, int>;
    std::map<PoolKey, std::size_t> pool_index;
    std::vector<PhysicalDevicePool> pools;
    std::map<std::uint32_t, std::size_t> node_host_available;
    std::map<std::uint32_t, std::size_t> node_fixed_host;
    const auto mandatory = cluster_host_reservations(raw_offers, coordinator);

    if (charge_control_directory) {
        node_host_available[coordinator.node] = coordinator.available_host_bytes;
    }

    for (std::size_t r = 0; r < raw_offers.size(); ++r) {
        auto offer = raw_offers[r];
        offer.node = node_for(offer, r);
        if (!offer.physical_placement_set) offer.device = 0;
        auto geometry       = describe_context_pool(offer.plan.records);
        const auto minimum  = minimum_bytes(offer.plan, geometry);
        const auto fixed    = checked_add(offer.plan.fixed_device_bytes, offer.graph_reserve_bytes);
        const auto raw_cap  = offer.maximum_runtime_bytes > offer.plan.fixed_device_bytes
                                  ? offer.maximum_runtime_bytes - offer.plan.fixed_device_bytes
                                  : 0;
        const auto pool_cap = offer.available_bytes > fixed ? offer.available_bytes - fixed : 0;
        auto max_back       = std::min(raw_cap, pool_cap);
        if (policy.mode == ContextCapacityMode::Automatic &&
            purpose == EnginePurpose::CausalScoring)
            max_back = std::min(
                max_back, scoring_context_capacity_limit(minimum, offer.plan.reference.device_bytes,
                                                         geometry.block_bytes));
        const auto selected =
            policy.mode == ContextCapacityMode::Explicit ? policy.explicit_bytes : minimum;
        if (selected < minimum || selected > max_back)
            insufficient(r, offer.plan, geometry, std::min(selected, max_back));

        auto [host, first_sample] =
            node_host_available.try_emplace(offer.node, offer.available_host_bytes);
        if (!first_sample) host->second = std::min(host->second, offer.available_host_bytes);
        if (!offer.shared_physical_memory) {
            node_fixed_host[offer.node] =
                checked_add(node_fixed_host[offer.node], offer.plan.fixed_host_bytes);
        }

        const PoolKey key{offer.node, offer.shared_physical_memory ? -1 : offer.device};
        auto [pit, inserted] = pool_index.emplace(key, pools.size());
        if (inserted) {
            PhysicalDevicePool pool;
            pool.node                   = offer.node;
            pool.device                 = key.second;
            pool.shared_physical_memory = offer.shared_physical_memory;
            pool.is_coordinator_uma = charge_control_directory && offer.shared_physical_memory &&
                                      offer.node == coordinator.node;
            pool.budget_bytes = offer.available_bytes;
            if (pool.is_coordinator_uma)
                pool.budget_bytes =
                    std::min(pool.budget_bytes, coordinator.available_unified_bytes);
            pools.push_back(std::move(pool));
        } else {
            pools[pit->second].budget_bytes =
                std::min(pools[pit->second].budget_bytes, offer.available_bytes);
        }
        pools[pit->second].ranks.push_back(r);
        owners.push_back({std::move(offer), std::move(geometry), minimum, max_back, selected});
    }

    std::map<std::uint32_t, std::size_t> node_host_remaining;
    for (const auto& [node, available] : node_host_available) {
        const auto reserved = checked_add(mandatory.by_node.at(node), node_fixed_host[node]);
        if (reserved > available)
            throw ContextBudgetError("Node " + std::to_string(node) +
                                     " Host reservations exceed available RAM");
        node_host_remaining[node] = available - reserved;
    }
    for (auto& pool : pools) {
        if (!pool.shared_physical_memory) continue;
        const auto reserved = node_host_available.at(pool.node) - node_host_remaining.at(pool.node);
        if (reserved > pool.budget_bytes)
            throw ContextBudgetError("Node " + std::to_string(pool.node) +
                                     " reservations exceed available UMA memory");
        pool.budget_bytes =
            std::min(pool.budget_bytes - reserved, node_host_remaining.at(pool.node));
    }

    // Check initial (minimum or explicit) feasibility on every shared device/UMA pool.
    const auto pool_used = [&](const PhysicalDevicePool& pool) {
        std::size_t sum = 0;
        for (auto r : pool.ranks)
            sum = checked_add(sum, owners[r].device_consumption(owners[r].selected));
        return sum;
    };

    for (const auto& pool : pools) {
        const auto used = pool_used(pool);
        if (used > pool.budget_bytes) {
            const auto r      = pool.ranks.back();
            const auto fixed  = checked_add(owners[r].offer.plan.fixed_device_bytes,
                                            owners[r].offer.graph_reserve_bytes);
            const auto others = used - owners[r].device_consumption(owners[r].selected);
            const auto avail =
                pool.budget_bytes > others + fixed ? pool.budget_bytes - others - fixed : 0;
            insufficient(r, owners[r].offer.plan, owners[r].geometry, avail);
        }
    }

    std::size_t metadata_total = 0;
    for (const auto& owner : owners)
        metadata_total = checked_add(metadata_total, owner.metadata(owner.selected));

    std::optional<std::size_t> coord_uma_pool;
    for (std::size_t i = 0; i < pools.size(); ++i)
        if (pools[i].is_coordinator_uma) coord_uma_pool = i;

    const auto coordinator_metadata_limit = [&]() -> std::size_t {
        if (!charge_control_directory) return SIZE_MAX;
        if (coord_uma_pool) {
            const auto& pool = pools[*coord_uma_pool];
            const auto used  = pool_used(pool);
            return used <= pool.budget_bytes
                       ? std::min(pool.budget_bytes - used, coordinator.maximum_directory_bytes)
                       : 0;
        }
        if (coordinator.shared_physical_memory) {
            const auto reserved =
                node_host_available.at(coordinator.node) - node_host_remaining.at(coordinator.node);
            return std::min({coordinator.available_unified_bytes - reserved,
                             node_host_remaining.at(coordinator.node),
                             coordinator.maximum_directory_bytes});
        }
        return std::min(node_host_remaining.at(coordinator.node),
                        coordinator.maximum_directory_bytes);
    };

    if (charge_control_directory && coordinator.shared_physical_memory && !coord_uma_pool &&
        node_host_available.at(coordinator.node) - node_host_remaining.at(coordinator.node) >
            coordinator.available_unified_bytes)
        throw ContextBudgetError("Coordinator Host reservations exceed available UMA memory");

    if (metadata_total > coordinator_metadata_limit()) {
        throw ContextBudgetError("Coordinator node " + std::to_string(coordinator.node) +
                                 " directory metadata (" + std::to_string(metadata_total) +
                                 " bytes) exceeds its available memory (" +
                                 std::to_string(coordinator_metadata_limit()) + " bytes)");
    }

    if (policy.mode == ContextCapacityMode::Automatic) {
        // Expand non-coordinator-UMA pools first so remote backings grow independently without
        // starving any rank's minimum directory metadata, then expand coordinator's UMA pool.
        std::vector<std::size_t> order;
        order.reserve(pools.size());
        for (std::size_t i = 0; i < pools.size(); ++i)
            if (!pools[i].is_coordinator_uma) order.push_back(i);
        if (coord_uma_pool) order.push_back(*coord_uma_pool);

        for (const auto idx : order) {
            const auto& pool      = pools[idx];
            std::size_t fixed_sum = 0;
            std::size_t old_meta  = 0;
            std::size_t high_cap  = 0;
            for (auto r : pool.ranks) {
                fixed_sum =
                    checked_add(fixed_sum, checked_add(owners[r].offer.plan.fixed_device_bytes,
                                                       owners[r].offer.graph_reserve_bytes));
                old_meta = checked_add(old_meta, owners[r].metadata(owners[r].selected));
                high_cap = std::max(high_cap, owners[r].max_back / owners[r].geometry.block_bytes *
                                                  owners[r].geometry.block_bytes);
            }
            const std::size_t other_meta = metadata_total - old_meta;

            const auto candidate_for = [&](std::size_t r, std::size_t byte_ceiling) {
                const auto& o     = owners[r];
                const auto capped = std::min(byte_ceiling, o.max_back);
                const auto blocks = capped / o.geometry.block_bytes;
                return std::max(o.minimum, blocks * o.geometry.block_bytes);
            };

            const auto fits_ceiling = [&](std::size_t byte_ceiling) {
                auto device_remaining = pool.budget_bytes - fixed_sum;
                std::size_t pool_meta = 0;
                for (auto r : pool.ranks) {
                    const auto b = candidate_for(r, byte_ceiling);
                    if (b > device_remaining) return false;
                    device_remaining -= b;
                    pool_meta = checked_add(pool_meta, owners[r].metadata(b));
                }
                if (!charge_control_directory) return true;
                const auto next_meta = checked_add(other_meta, pool_meta);
                if (pool.is_coordinator_uma)
                    return next_meta <= device_remaining &&
                           next_meta <= coordinator.maximum_directory_bytes;
                return next_meta <= coordinator_metadata_limit();
            };

            // A zero ceiling reproduces every rank's independently verified minimum.
            const auto best_ceiling = largest(0, high_cap, fits_ceiling);

            metadata_total = other_meta;
            for (auto r : pool.ranks) {
                owners[r].selected = candidate_for(r, best_ceiling);
                metadata_total =
                    checked_add(metadata_total, owners[r].metadata(owners[r].selected));
            }

            // Distribute any remaining whole blocks round-robin across co-located ranks in pool.
            if (pool.ranks.size() > 1) {
                bool grew = true;
                while (grew) {
                    grew = false;
                    for (auto r : pool.ranks) {
                        auto& o = owners[r];
                        if (o.max_back - o.selected < o.geometry.block_bytes) continue;
                        const auto used = pool_used(pool);
                        if (o.geometry.block_bytes > pool.budget_bytes - used) continue;
                        const auto next           = o.selected + o.geometry.block_bytes;
                        const auto next_pool_used = used + o.geometry.block_bytes;
                        const auto next_meta =
                            checked_add(metadata_total - o.metadata(o.selected), o.metadata(next));
                        if (charge_control_directory) {
                            if (pool.is_coordinator_uma) {
                                if (next_meta > pool.budget_bytes - next_pool_used ||
                                    next_meta > coordinator.maximum_directory_bytes)
                                    continue;
                            } else if (next_meta > coordinator_metadata_limit()) {
                                continue;
                            }
                        }
                        o.selected     = next;
                        metadata_total = next_meta;
                        grew           = true;
                    }
                }
            }
        }
    }

    // Verify the solved inequalities before exposing capacities to the allocation phase.
    for (const auto& pool : pools)
        if (pool_used(pool) > pool.budget_bytes)
            throw std::logic_error("solved Context capacities exceed the physical device pool");
    for (const auto& [node, available] : node_host_remaining) {
        const auto metadata =
            charge_control_directory && node == coordinator.node ? metadata_total : 0;
        if (metadata > available)
            throw std::logic_error("solved Context metadata exceeds the node Host budget");
    }
    if (metadata_total > coordinator_metadata_limit())
        throw std::logic_error("solved Context metadata exceeds the coordinator budget");
    for (const auto& owner : owners)
        if (checked_add(owner.offer.plan.fixed_device_bytes, owner.selected) >
            owner.offer.maximum_runtime_bytes)
            throw std::logic_error("solved Context capacity exceeds the rank runtime limit");

    // Assign control_reservation_bytes to the first local rank on coordinator.node (if any) for
    // per-rank budget reporting; pure remote clusters still validated metadata_total above.
    std::optional<std::size_t> report_control_rank;
    if (charge_control_directory) {
        for (std::size_t r = 0; r < owners.size(); ++r) {
            if (owners[r].offer.node == coordinator.node) {
                report_control_rank = r;
                break;
            }
        }
    }

    // Map each rank back to its physical pool to report exact pool slack.
    std::vector<std::size_t> rank_pool(owners.size(), 0);
    for (std::size_t p = 0; p < pools.size(); ++p)
        for (auto r : pools[p].ranks) rank_pool[r] = p;

    std::vector<ContextCapacityResolution> result;
    result.reserve(owners.size());
    for (std::size_t r = 0; r < owners.size(); ++r) {
        const auto& owner        = owners[r];
        const auto& pool         = pools[rank_pool[r]];
        const auto runtime_bytes = checked_add(owner.offer.plan.fixed_device_bytes, owner.selected);
        const auto control_bytes = report_control_rank == r ? metadata_total : 0;
        const auto total_dev     = pool_used(pool);
        const auto uma_meta      = pool.is_coordinator_uma ? metadata_total : 0;
        const auto pool_slack    = pool.budget_bytes >= total_dev + uma_meta
                                       ? pool.budget_bytes - total_dev - uma_meta
                                       : 0;
        const auto rank_avail    = runtime_bytes + pool_slack;
        result.push_back({
            .mode                 = policy.mode,
            .device_bytes         = owner.selected,
            .minimum_device_bytes = owner.minimum,
            .block_bytes          = owner.geometry.block_bytes,
            .minimum_runtime_reservation_bytes =
                checked_add(owner.offer.plan.fixed_device_bytes, owner.minimum),
            .runtime_reservation_bytes     = runtime_bytes,
            .available_after_weights_bytes = rank_avail,
            .planned_slack_bytes           = pool_slack,
            .metadata_budget_bytes         = owner.metadata(owner.selected),
            .control_reservation_bytes     = control_bytes,
            .reference                     = owner.offer.plan.reference,
        });
    }
    return result;
}

} // namespace

std::size_t scoring_context_capacity_limit(std::size_t minimum, std::size_t reference,
                                           std::size_t block_bytes) {
    return std::max(minimum, reference / block_bytes * block_bytes);
}

std::size_t reduced_runtime_limit(const ContextCapacityResolution& capacity,
                                  std::size_t backing_reduction) {
    return capacity.runtime_reservation_bytes -
           std::min(backing_reduction, capacity.device_bytes - capacity.minimum_device_bytes);
}

ClusterHostReservations
cluster_host_reservations(std::span<const RankCapacityOffer> offers,
                          const CoordinatorMemoryOffer& coordinator,
                          std::span<const ContextCapacityResolution> capacities) {
    ClusterHostReservations ledger;
    for (const auto& capacity : capacities)
        ledger.coordinator_directory_bytes =
            checked_add(ledger.coordinator_directory_bytes, capacity.metadata_budget_bytes);
    ledger.by_node[coordinator.node] =
        checked_add(coordinator.frontend_host_bytes, ledger.coordinator_directory_bytes);
    for (std::size_t r = 0; r < offers.size(); ++r) {
        auto& reserved = ledger.by_node[node_for(offers[r], r)];
        reserved       = checked_add(reserved, offers[r].host_reservation_bytes);
    }
    return ledger;
}

ContextCapacity ContextCapacityPlan::minimum_capacity() const {
    const auto geometry = describe_context_pool(records);
    return {minimum_bytes(*this, geometry)};
}

std::size_t ContextCapacityPlan::reservation_bytes(ContextCapacity capacity) const {
    if (capacity.device_bytes < minimum_capacity().device_bytes)
        throw std::invalid_argument("Context backing cannot hold the startup working sets");
    return checked_add(fixed_device_bytes, capacity.device_bytes);
}

std::size_t ContextCapacityPlan::control_bytes(ContextCapacity capacity) const {
    const auto geometry = describe_context_pool(records);
    return runtime::control_bytes(metadata, geometry, capacity.device_bytes);
}

std::vector<ContextCapacityResolution>
resolve_cluster_context_capacity(const ContextCapacityPolicy& policy,
                                 std::span<const RankCapacityOffer> offers,
                                 const CoordinatorMemoryOffer& coordinator, EnginePurpose purpose) {
    return solve_capacities(policy, offers, coordinator, true, purpose);
}

std::vector<ContextCapacityResolution>
resolve_group_context_capacity(const ContextCapacityPolicy& policy,
                               std::span<const RankCapacityOffer> offers,
                               std::optional<std::size_t> control_memory_owner) {
    if (offers.empty()) throw std::invalid_argument("Context capacity requires memory owners");
    if (control_memory_owner && *control_memory_owner >= offers.size())
        throw std::invalid_argument("Context directory memory owner is outside the rank group");
    CoordinatorMemoryOffer coordinator;
    if (control_memory_owner) {
        const auto& owner                   = offers[*control_memory_owner];
        coordinator.node                    = owner.physical_placement_set
                                                  ? owner.node
                                                  : static_cast<std::uint32_t>(*control_memory_owner);
        coordinator.available_host_bytes    = owner.available_host_bytes;
        coordinator.available_unified_bytes = owner.available_bytes;
        coordinator.shared_physical_memory  = owner.shared_physical_memory;
    }
    return solve_capacities(policy, offers, coordinator, control_memory_owner.has_value(),
                            EnginePurpose::Generation);
}

std::uint64_t
estimate_context_token_capacity(std::span<const ContextCapacityResolution> capacities) {
    if (capacities.empty())
        throw std::invalid_argument("Context capacity estimate requires rank capacities");
    const auto& reference = capacities.front().reference;
    std::uint64_t tokens  = UINT64_MAX;
    for (const auto& capacity : capacities) {
        const auto& local = capacity.reference;
        if (!capacity.block_bytes || !local.device_bytes || !local.tokens_per_request ||
            !local.request_count || local.tokens_per_request != reference.tokens_per_request ||
            local.request_count != reference.request_count)
            throw std::invalid_argument(
                "Context capacity estimate requires matching model references");
        const auto usable   = capacity.device_bytes / capacity.block_bytes * capacity.block_bytes;
        const auto estimate = static_cast<unsigned __int128>(usable) * local.request_count *
                              local.tokens_per_request / local.device_bytes;
        if (estimate < tokens) tokens = static_cast<std::uint64_t>(estimate);
    }
    return tokens;
}

ContextCapacityResolution
resolve_context_capacity(const ContextCapacityPolicy& policy, const ContextCapacityPlan& plan,
                         std::size_t available, std::size_t available_host_bytes,
                         bool shared_physical_memory, bool owns_control_memory,
                         EnginePurpose purpose) {
    // The single-rank UMA allowance already reflects Host reservations made by its startup owner.
    const std::array offers{
        RankCapacityOffer{plan, available, shared_physical_memory ? SIZE_MAX : available_host_bytes,
                          shared_physical_memory}};
    const CoordinatorMemoryOffer coordinator{0, offers.front().available_host_bytes, available,
                                             shared_physical_memory};
    return solve_capacities(policy, offers, coordinator, owns_control_memory, purpose).front();
}

} // namespace ninfer::runtime
