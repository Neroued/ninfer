#include "runtime/engine/context_capacity.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>

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
    const RankCapacityOffer& offer;
    ContextPoolGeometry geometry;
    std::size_t minimum;
    std::size_t usable;
    std::size_t selected;

    [[nodiscard]] std::size_t metadata(std::size_t bytes) const {
        return control_bytes(offer.plan.metadata, geometry, bytes);
    }

    [[nodiscard]] std::size_t metadata_capacity(std::size_t backing) const {
        return offer.shared_physical_memory
                   ? usable - backing
                   : offer.available_host_bytes - offer.plan.fixed_host_bytes;
    }
};

[[noreturn]] void insufficient(std::size_t rank, const ContextCapacityPlan& plan,
                               const ContextPoolGeometry& geometry, std::size_t available) {
    for (const auto& set : plan.working_sets) {
        const auto required = geometry.required_bytes(set.records);
        if (required > available)
            throw std::invalid_argument("Context rank " + std::to_string(rank) +
                                        " cannot hold startup working set '" + set.name +
                                        "': requires " + std::to_string(required) +
                                        " backing bytes, available " + std::to_string(available));
    }
    throw std::invalid_argument("Context rank " + std::to_string(rank) +
                                " capacity and directory metadata exceed the memory budget");
}

} // namespace

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
resolve_group_context_capacity(const ContextCapacityPolicy& policy,
                               std::span<const RankCapacityOffer> offers,
                               std::optional<std::size_t> control_memory_owner) {
    validate_policy(policy);
    if (offers.empty()) throw std::invalid_argument("Context capacity requires memory owners");
    if (control_memory_owner && *control_memory_owner >= offers.size())
        throw std::invalid_argument("Context directory memory owner is outside the rank group");

    std::vector<CapacityOwner> owners;
    owners.reserve(offers.size());
    std::size_t metadata_total = 0;
    for (std::size_t rank = 0; rank < offers.size(); ++rank) {
        const auto& offer  = offers[rank];
        auto geometry      = describe_context_pool(offer.plan.records);
        const auto minimum = minimum_bytes(offer.plan, geometry);
        const auto fixed   = offer.plan.fixed_device_bytes;
        const auto usable  = offer.available_bytes > fixed ? offer.available_bytes - fixed : 0;
        const auto selected =
            policy.mode == ContextCapacityMode::Explicit ? policy.explicit_bytes : minimum;
        if (selected < minimum || selected > usable)
            insufficient(rank, offer.plan, geometry, std::min(selected, usable));
        if (!offer.shared_physical_memory &&
            offer.plan.fixed_host_bytes > offer.available_host_bytes)
            throw std::invalid_argument("Context rank " + std::to_string(rank) +
                                        " fixed Host resources exceed available RAM");
        owners.push_back({offer, std::move(geometry), minimum, usable, selected});
        metadata_total = checked_add(metadata_total, owners.back().metadata(selected));
    }
    if (control_memory_owner) {
        const auto& control = owners[*control_memory_owner];
        if (metadata_total > control.metadata_capacity(control.selected))
            throw std::invalid_argument("Context rank " + std::to_string(*control_memory_owner) +
                                        " directory metadata exceeds its available memory");
    }

    if (policy.mode == ContextCapacityMode::Automatic) {
        // Charge every directory to its actual physical owner. Remote backings may grow
        // independently, but cannot consume the metadata needed by another rank's minimum.
        for (std::size_t rank = 0; rank < owners.size(); ++rank) {
            if (control_memory_owner == rank) continue;
            auto& owner       = owners[rank];
            const auto before = owner.metadata(owner.selected);
            const auto blocks =
                largest(owner.minimum / owner.geometry.block_bytes,
                        owner.usable / owner.geometry.block_bytes, [&](std::size_t count) {
                            if (!control_memory_owner) return true;
                            const auto& control = owners[*control_memory_owner];
                            const auto next  = owner.metadata(count * owner.geometry.block_bytes);
                            const auto other = metadata_total - before;
                            const auto remaining = control.metadata_capacity(control.selected);
                            return other <= remaining && next <= remaining - other;
                        });
            owner.selected = blocks * owner.geometry.block_bytes;
            metadata_total = checked_add(metadata_total - before, owner.metadata(owner.selected));
        }
        if (control_memory_owner) {
            auto& owner      = owners[*control_memory_owner];
            const auto other = metadata_total - owner.metadata(owner.selected);
            const auto blocks =
                largest(owner.minimum / owner.geometry.block_bytes,
                        owner.usable / owner.geometry.block_bytes, [&](std::size_t count) {
                            const auto bytes     = count * owner.geometry.block_bytes;
                            const auto remaining = owner.metadata_capacity(bytes);
                            return other <= remaining && owner.metadata(bytes) <= remaining - other;
                        });
            owner.selected = blocks * owner.geometry.block_bytes;
            metadata_total = checked_add(other, owner.metadata(owner.selected));
        }
    }

    std::vector<ContextCapacityResolution> result;
    result.reserve(owners.size());
    for (std::size_t rank = 0; rank < owners.size(); ++rank) {
        const auto& owner        = owners[rank];
        const auto runtime_bytes = checked_add(owner.offer.plan.fixed_device_bytes, owner.selected);
        const auto control_bytes = control_memory_owner == rank ? metadata_total : 0;
        const auto device_control = owner.offer.shared_physical_memory ? control_bytes : 0;
        result.push_back({
            .mode                 = policy.mode,
            .device_bytes         = owner.selected,
            .minimum_device_bytes = owner.minimum,
            .block_bytes          = owner.geometry.block_bytes,
            .minimum_runtime_reservation_bytes =
                checked_add(owner.offer.plan.fixed_device_bytes, owner.minimum),
            .runtime_reservation_bytes     = runtime_bytes,
            .available_after_weights_bytes = owner.offer.available_bytes,
            .planned_slack_bytes   = owner.offer.available_bytes - runtime_bytes - device_control,
            .metadata_budget_bytes = owner.metadata(owner.selected),
            .control_reservation_bytes = control_bytes,
            .reference                 = owner.offer.plan.reference,
        });
    }
    return result;
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
                         bool shared_physical_memory, bool owns_control_memory) {
    const std::array offers{
        RankCapacityOffer{plan, available, available_host_bytes, shared_physical_memory}};
    return resolve_group_context_capacity(
               policy, offers, owns_control_memory ? std::optional<std::size_t>{0} : std::nullopt)
        .front();
}

} // namespace ninfer::runtime
