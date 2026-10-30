#pragma once

#include "ninfer/kv_cache_format.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninfer {

enum class ContextCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

// Vision fields describe subsets of the instance's owned allocations.
struct VisionMemorySummary {
    std::size_t budget_bytes            = 0;
    std::uint32_t max_window_features   = 0;
    std::size_t encoder_workspace_bytes = 0;
    std::size_t output_capacity_bytes   = 0;
    std::size_t input_staging_bytes     = 0;
    std::size_t output_active_bytes     = 0;
    std::size_t output_peak_bytes       = 0;
};

struct MoeMemorySummary {
    std::size_t host_weights_bytes    = 0;
    std::size_t registered_host_bytes = 0;
    std::size_t cache_bytes           = 0;
    std::size_t fixed_bytes           = 0;
    std::size_t window_bytes          = 0;
    std::uint32_t window_count        = 0;
    std::size_t used_bytes            = 0;
    std::size_t target_bytes          = 0;
    std::size_t ready_bytes           = 0;
    std::uint32_t cpu_workers         = 0;
    std::vector<std::uint32_t> fixed_experts_per_layer;
    std::vector<std::uint32_t> target_experts_per_layer;
    std::vector<std::uint32_t> ready_experts_per_layer;
    std::size_t bridge_device_bytes = 0;
    std::size_t bridge_pinned_bytes = 0;
    std::size_t cpu_workspace_bytes = 0;
    std::uint64_t uploaded_bytes    = 0;
    std::uint64_t prefetched_bytes  = 0;
};

struct GraphMemorySummary {
    std::size_t rebuild_estimate_bytes = 0;
    // Signed physical-capacity deltas. Rebuild estimates come from an actual second capture.
    std::int64_t warmup_increment_bytes      = 0;
    std::int64_t preparation_increment_bytes = 0;
    std::size_t preparation_peak_bytes       = 0;
    std::int64_t calibration_increment_bytes = 0;
    std::size_t calibration_peak_bytes       = 0;
    std::int64_t rebuild_increment_bytes     = 0;
};

// Owned allocations. Registered ranges, tensor views and layout subregions are not allocations.
struct MemoryAllocations {
    std::uint64_t device_weights_bytes     = 0;
    std::uint64_t host_weights_bytes       = 0;
    std::uint64_t registered_weights_bytes = 0;
    std::uint64_t persistent_device_bytes  = 0;
    std::uint64_t context_device_bytes     = 0;
    std::uint64_t workspace_device_bytes   = 0;
    std::uint64_t io_pinned_bytes          = 0;
    std::uint64_t host_context_bytes       = 0;
    std::uint64_t ple_device_bytes         = 0;
    std::uint64_t ple_pinned_bytes         = 0;
    std::uint64_t ple_host_bytes           = 0;
    std::uint64_t expert_device_bytes      = 0;
    std::uint64_t expert_pinned_bytes      = 0;
    std::uint64_t expert_host_bytes        = 0;

    [[nodiscard]] std::uint64_t device_bytes() const noexcept {
        return device_weights_bytes + persistent_device_bytes + context_device_bytes +
               workspace_device_bytes + ple_device_bytes + expert_device_bytes;
    }

    [[nodiscard]] std::uint64_t host_bytes() const noexcept {
        return host_weights_bytes + io_pinned_bytes + host_context_bytes + ple_pinned_bytes +
               ple_host_bytes + expert_pinned_bytes + expert_host_bytes;
    }
};

struct WorkspaceUsage {
    // Host arena cursor, not the lifetime of asynchronous device consumers.
    std::uint64_t allocation_cursor_bytes = 0;
    std::uint64_t required_peak_bytes     = 0;
    std::uint64_t measurement_peak_bytes  = 0;
};

struct PleReadStats {
    std::uint64_t rows = 0, cache_hits = 0, shared_rows = 0, shared_pages = 0;
    std::uint64_t reads = 0, read_bytes = 0, uploaded_bytes = 0;
};

struct RankMemoryUsage {
    MemoryAllocations allocations;
    WorkspaceUsage workspace;
    std::optional<VisionMemorySummary> vision;
    std::optional<MoeMemorySummary> experts;
    std::optional<PleReadStats> ple;
};

struct PhysicalMemorySample {
    std::uint64_t cuda_total_bytes = 0, cuda_free_bytes = 0;
    std::uint64_t system_total_bytes = 0, system_available_bytes = 0;
    bool shared_physical_memory = false;
    std::uint64_t sampled_ns    = 0;
};

struct RankMemoryBudget {
    std::uint64_t limit_bytes                       = 0;
    std::uint64_t runtime_reservation_bytes         = 0;
    std::uint64_t minimum_runtime_reservation_bytes = 0;
    std::uint64_t context_capacity_bytes            = 0;
    std::uint64_t minimum_context_capacity_bytes    = 0;
    std::uint64_t context_control_budget_bytes      = 0;
    std::uint64_t available_after_weights_bytes     = 0;
    std::uint64_t available_after_startup_bytes     = 0;
    // Planning headroom may be consumed during startup; Host reservation must remain available.
    std::uint64_t headroom_bytes         = 0;
    std::uint64_t host_reservation_bytes = 0;
    std::uint64_t planned_slack_bytes    = 0;
    std::int64_t retained_bytes          = 0;
    std::optional<GraphMemorySummary> graphs;
};

enum class ContextMemoryDomain : std::uint8_t { Device, Host };

struct ContextPoolCapacity {
    std::uint32_t rank           = 0;
    ContextMemoryDomain domain   = ContextMemoryDomain::Device;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t block_bytes    = 0;
    friend bool operator==(const ContextPoolCapacity&, const ContextPoolCapacity&) = default;
};

// These disjoint occupancies account for the complete backing. Relocation destinations are
// reserved until publication; the old records remain live or pending retirement until then.
struct ContextPoolUsage {
    std::uint32_t rank                     = 0;
    ContextMemoryDomain domain             = ContextMemoryDomain::Device;
    std::uint64_t capacity_bytes           = 0;
    std::uint64_t block_bytes              = 0;
    std::uint64_t live_bytes               = 0;
    std::uint64_t reserved_bytes           = 0;
    std::uint64_t pending_retirement_bytes = 0;
    // Relocation is a subset of reserved bytes; future reservations promise positions
    // released by maintenance and do not occupy those positions yet.
    std::uint64_t relocation_bytes                                           = 0;
    std::uint64_t future_reserved_bytes                                      = 0;
    std::uint64_t padding_bytes                                              = 0;
    std::uint64_t assigned_free_bytes                                        = 0;
    std::uint64_t unassigned_free_bytes                                      = 0;
    std::uint64_t tail_bytes                                                 = 0;
    std::uint64_t peak_occupied_bytes                                        = 0;
    friend bool operator==(const ContextPoolUsage&, const ContextPoolUsage&) = default;
};

struct ContextMetadataUsage {
    std::uint64_t budget_bytes                                                       = 0;
    std::uint64_t allocated_bytes                                                    = 0;
    std::uint64_t reserved_bytes                                                     = 0;
    friend bool operator==(const ContextMetadataUsage&, const ContextMetadataUsage&) = default;
};

// Materialized physical records, counted once per replica independently of logical references.
// Pool reservations, padding and free capacity remain in ContextPoolUsage.
struct ContextResourceUsage {
    std::uint32_t rank         = 0;
    ContextMemoryDomain domain = ContextMemoryDomain::Device;
    std::optional<std::uint16_t> history_pool;
    std::uint64_t live_records = 0, live_bytes = 0;
    std::uint64_t pending_retirement_records = 0, pending_retirement_bytes = 0;
};

struct PrefixCacheCounters {
    std::uint64_t retained = 0, published = 0, released = 0, reclaimed = 0;
};

// Advertised checkpoint identities, independently of their physical replica/reference count.
struct PrefixCacheStatistics {
    PrefixCacheCounters input_replay, long_anchor, shared_prefix, continuation;
};

struct MediaCacheSummary {
    std::size_t capacity_bytes       = 0;
    std::size_t live_capacity_bytes  = 0;
    std::size_t retained_bytes       = 0;
    std::size_t live_bytes           = 0;
    std::size_t entries              = 0;
    std::size_t inflight             = 0;
    std::size_t queued_tasks         = 0;
    std::size_t active_tasks         = 0;
    std::uint32_t preprocess_threads = 0;
    std::uint64_t hits               = 0;
    std::uint64_t misses             = 0;
    std::uint64_t singleflight_waits = 0;
    std::uint64_t evictions          = 0;
    std::uint64_t oversize_bypasses  = 0;
};

} // namespace ninfer
