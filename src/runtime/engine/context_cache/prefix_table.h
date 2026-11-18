#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace ninfer::runtime {

// Normalized per-checkpoint facts for the prefix table and on-demand releases. Model sessions
// supply these while no execution is in flight; the policies here stay model-agnostic and
// testable.
struct PrefixCheckpointRecord {
    ninfer::PrefixCacheKey key;
    bool leased = false;
    std::uint32_t state_slots = 0;
    std::array<std::uint32_t, ninfer::kMaximumKvPools> kv_page_groups{};
    std::size_t host_bytes = 0;
    bool reused = false; // This checkpoint has had repeat demand.
    std::uint64_t last_demand = 0; // Latest logical demand tick for this checkpoint.
};

template <class Value>
[[nodiscard]] constexpr Value saturating_add(Value current, Value increment) noexcept {
    return increment > std::numeric_limits<Value>::max() - current
               ? std::numeric_limits<Value>::max()
               : current + increment;
}

// Evictable resources of one record: device KV bytes (page groups x device_page_bytes per
// pool, KvPoolDescriptor), host bytes (host pages x host_page_stride, as reported by the
// session), and state images (state_slots x state_image_bytes). Zero is a valid total for a
// fully shared prefix that holds no state slot and no KV.
[[nodiscard]] inline std::uint64_t
record_evictable_bytes(const PrefixCheckpointRecord& record,
                       std::span<const std::size_t> device_page_bytes,
                       std::size_t state_image_bytes) noexcept {
    std::uint64_t bytes = record.host_bytes;
    const std::size_t pools = std::min(record.kv_page_groups.size(), device_page_bytes.size());
    for (std::size_t pool = 0; pool < pools; ++pool) {
        const std::uint64_t pool_bytes =
            static_cast<std::uint64_t>(record.kv_page_groups[pool]) * device_page_bytes[pool];
        bytes = saturating_add(bytes, pool_bytes);
    }
    const std::uint64_t state_bytes =
        static_cast<std::uint64_t>(record.state_slots) * state_image_bytes;
    return saturating_add(bytes, state_bytes);
}

// Groups records by prefix identity, preserving walk (reclaim) order, and aggregates the
// evictable resources of every resident checkpoint that shares a key.
[[nodiscard]] inline ninfer::PrefixCacheTable
build_prefix_entries(std::span<const PrefixCheckpointRecord> records,
                     std::span<const std::size_t> device_page_bytes,
                     std::size_t state_image_bytes) {
    ninfer::PrefixCacheTable table;
    for (const auto& record : records) {
        auto found = std::find_if(table.prefixes.begin(), table.prefixes.end(),
                                  [&](const auto& entry) { return entry.key == record.key; });
        auto& entry =
            found == table.prefixes.end()
                ? table.prefixes.emplace_back(
                      ninfer::PrefixCacheEntry{.key = record.key, .tokens = record.key.frontier})
                : *found;
        entry.leased      = entry.leased || record.leased;
        entry.checkpoints += 1;
        entry.reused      = entry.reused || record.reused;
        entry.last_demand = std::max(entry.last_demand, record.last_demand);
        entry.state_slots = saturating_add(entry.state_slots, record.state_slots);
        entry.host_bytes  = saturating_add(entry.host_bytes, record.host_bytes);
        for (std::size_t pool = 0;
             pool < std::min(entry.kv_page_groups.size(), record.kv_page_groups.size()); ++pool) {
            entry.kv_page_groups[pool] =
                saturating_add(entry.kv_page_groups[pool], record.kv_page_groups[pool]);
        }
        entry.evictable_bytes = saturating_add(
            entry.evictable_bytes,
            record_evictable_bytes(record, device_page_bytes, state_image_bytes));
    }
    return table;
}

// The engine is idle for on-demand prefix releases when nothing is pending, no request is
// resident or materializing (resident_empty), nothing is paused, and no context operation is in
// flight. This mirrors the worker loop's idle predicate: on-demand releases never compete with
// in-flight requests, including requests still queued for admission.
[[nodiscard]] inline bool
prefix_cache_idle(bool has_pending, bool resident_empty, bool has_paused,
                  bool has_context_operation) noexcept {
    return !has_pending && resident_empty && !has_paused && !has_context_operation;
}

// Decides which resident checkpoints to release for each requested prefix, in request order.
// max_bytes bounds the release (0 means unbounded): processing stops once the expected total
// reaches the cap and the remaining requests stay resident.
struct PrefixEvictionPlan {
    struct KeyPlan {
        enum class Action : std::uint8_t { Evict, Leased, NotFound };
        ninfer::PrefixCacheKey key;
        Action action = Action::NotFound;
        std::vector<std::size_t> records; // Indices into the walked records, in release order.
        std::uint64_t expected_freed_bytes = 0;
        bool truncated = false; // The cap was reached inside this prefix.
    };

    std::vector<KeyPlan> keys;
    std::uint64_t expected_total_bytes = 0;
    bool stopped = false; // The cap was reached; later requests were not processed.
};

[[nodiscard]] inline PrefixEvictionPlan
plan_prefix_evictions(std::span<const PrefixCheckpointRecord> records,
                      std::span<const ninfer::PrefixCacheKey> selection, std::uint64_t max_bytes,
                      std::span<const std::size_t> device_page_bytes,
                      std::size_t state_image_bytes) {
    PrefixEvictionPlan plan;
    for (const auto& key : selection) {
        std::vector<std::size_t> group;
        bool leased = false;
        for (std::size_t index = 0; index < records.size(); ++index) {
            if (records[index].key != key) { continue; }
            group.push_back(index);
            leased = leased || records[index].leased;
        }
        if (group.empty()) {
            plan.keys.push_back(
                {.key = key, .action = PrefixEvictionPlan::KeyPlan::Action::NotFound});
            continue;
        }
        if (leased) {
            plan.keys.push_back(
                {.key = key, .action = PrefixEvictionPlan::KeyPlan::Action::Leased});
            continue;
        }
        PrefixEvictionPlan::KeyPlan planned{
            .key   = key,
            .action = PrefixEvictionPlan::KeyPlan::Action::Evict};
        for (const std::size_t index : group) {
            const auto bytes =
                record_evictable_bytes(records[index], device_page_bytes, state_image_bytes);
            if (max_bytes != 0 && plan.expected_total_bytes >= max_bytes) {
                planned.truncated = true;
                plan.stopped      = true;
                break;
            }
            planned.records.push_back(index);
            planned.expected_freed_bytes = saturating_add(planned.expected_freed_bytes, bytes);
            plan.expected_total_bytes    = saturating_add(plan.expected_total_bytes, bytes);
        }
        plan.keys.push_back(std::move(planned));
    }
    return plan;
}

} // namespace ninfer::runtime
