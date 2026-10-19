#pragma once

#include "ninfer/observability/resources.h"
#include "ninfer/observability/execution.h"

#include <filesystem>

namespace ninfer {

// Worker-owned monotonic nanosecond counters. Top-level Host phases are mutually exclusive;
// scheduler_yield_ns is scheduler yield/relock wall time and is intentionally excluded from their
// sum. Detail values are subsets of a top-level phase and must not be added to Host-active time
// again.
struct RuntimeHostWorkStats {
    std::uint64_t engine_boundary_ns      = 0;
    std::uint64_t session_call_ns         = 0;
    std::uint64_t executor_poll_ns        = 0;
    std::uint64_t engine_commit_output_ns = 0;
    std::uint64_t engine_maintenance_ns   = 0;
    std::uint64_t scheduler_yield_ns      = 0;
    std::uint64_t executor_flush_ns       = 0;

    std::uint64_t decode_host_ns             = 0;
    std::uint64_t decode_scheduler_yield_ns  = 0;
    std::uint64_t prefill_host_ns            = 0;
    std::uint64_t prefill_scheduler_yield_ns = 0;
    std::uint64_t control_host_ns            = 0;
    std::uint64_t control_scheduler_yield_ns = 0;
    std::uint64_t prefill_units              = 0;
    std::uint64_t control_units              = 0;

    std::uint64_t stats_publication_ns          = 0;
    std::uint64_t stats_publication_invocations = 0;
};

// Monotonic execution counters, boundary-consistent current gauges, and explicitly named last
// decision observations. Consumers derive interval counters by subtracting two snapshots.
struct RuntimeStats {
    RuntimeHostWorkStats host_work;
    // Full prompt counted once when initial binding succeeds; includes reused tokens.
    std::uint64_t prompt_tokens = 0;
    // All committed output tokens, including the first token and Engine-injected control tokens.
    std::uint64_t generated_tokens = 0;
    // Native verification work, using the same acceptance semantics as SpeculativeStats.
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    // Initial prompt tokens evaluated by prefill. Reused checkpoint-prefix tokens and replay
    // recomputation are excluded; replayed_tokens separately counts that additional model work.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds             = 0;
    std::uint64_t decode_row_rounds         = 0;
    std::uint32_t running_requests          = 0;
    std::uint32_t prefilling_requests       = 0;
    std::uint32_t decode_ready_requests     = 0;
    std::uint32_t waiting_requests          = 0;
    std::uint32_t paused_requests           = 0;
    std::uint32_t replaying_requests        = 0;
    std::uint32_t materializing_requests    = 0;
    std::uint32_t capture_pending_requests  = 0;
    std::uint32_t terminal_pending_requests = 0;
    std::uint64_t active_captures_completed = 0;
    std::uint64_t active_captures_aborted   = 0;
    std::uint64_t preemptions               = 0;
    std::uint64_t snapshot_restores         = 0;
    std::uint64_t replay_restores           = 0;
    std::uint64_t replayed_tokens           = 0;

    std::uint64_t root_selections               = 0;
    std::uint64_t checkpoint_selections         = 0;
    std::uint64_t reused_prompt_tokens          = 0;
    std::uint32_t last_selected_frontier_tokens = 0;

    std::uint64_t state_moves                 = 0;
    std::uint64_t state_forks                 = 0;
    std::uint64_t materialization_state_forks = 0;
    std::uint64_t state_restores              = 0;
    ContextTransferStats context_transfers;
    std::vector<ContextKvPoolUsage> kv_pools;

    std::uint64_t pressure_spill_pages        = 0;
    std::uint64_t partial_tail_cow_pages      = 0;
    std::uint32_t device_state_occupied_slots = 0;
    std::uint32_t host_state_occupied_slots   = 0;
    std::size_t host_kv_occupied_bytes        = 0;
    // Unified physical Host context occupancy. Reserved destinations are included in occupied.
    std::size_t host_context_occupied_bytes = 0;
    std::size_t host_context_reserved_bytes = 0;
    // Allocator lifetime high-water mark, including reserved transfer destinations.
    std::size_t host_context_peak_occupied_bytes = 0;
    double actual_context_transfer_seconds       = 0.0;
};

enum class ContextCostPresetSource : std::uint8_t {
    GenericDefault,
    CompiledDefault,
    External,
};

[[nodiscard]] inline constexpr const char*
context_cost_preset_source_name(ContextCostPresetSource source) noexcept {
    switch (source) {
    case ContextCostPresetSource::GenericDefault:
        return "generic-default";
    case ContextCostPresetSource::CompiledDefault:
        return "compiled-default";
    case ContextCostPresetSource::External:
        return "external";
    }
    return "unknown";
}

struct ContextCostSummary {
    ContextCostPresetSource transfer_source = ContextCostPresetSource::GenericDefault;
    ContextCostPresetSource prefill_source  = ContextCostPresetSource::GenericDefault;
    std::string hardware_class;
    std::string prefill_signature;
    std::filesystem::path preset_path;
};

struct LoadSummary {
    std::string architecture;
    std::string model_name;
    std::string cuda_sync_mode;
    std::vector<std::string> weight_formats;
    std::string prefill_signature;
    double load_seconds                  = 0.0;
    double upload_seconds                = 0.0;
    std::uint64_t artifact_bytes_read    = 0;
    std::uint64_t host_to_device_bytes   = 0;
    std::uint64_t max_rank_staging_bytes = 0;
    std::size_t device_object_count      = 0;
    std::size_t host_object_count        = 0;
    ContextCostSummary context_cost;
};

struct RankDescriptor {
    std::uint32_t rank = 0, node = 0;
    int device = 0;
    std::string hardware;
    bool shared_physical_memory = false;
};

struct EngineDescription {
    std::string architecture, model_name;
    std::vector<RankDescriptor> ranks;
    std::uint32_t max_context = 0, max_concurrency = 0, prefill_chunk = 0;
    std::uint32_t kv_capacity = 0, device_state_capacity = 0;
    std::uint64_t host_context_capacity_bytes = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::BFloat16;
    std::vector<ContextKvPoolUsage> kv_pools;
};

struct StartupReport {
    LoadSummary load;
    std::vector<RankMemoryBudget> ranks;
};

struct SnapshotStamp {
    std::uint64_t sequence = 0;
    // Nanoseconds on the producing process's monotonic clock. Cross-process timestamps are
    // never subtracted; a coordinator also records when each remote sample was received.
    std::uint64_t sampled_ns = 0;
};

struct RankSnapshot {
    std::string hardware;
    std::uint32_t rank = 0, node = 0;
    SnapshotStamp stamp;
    std::uint64_t received_ns = 0;
    RankMemoryUsage memory;
    RankHostStatistics host;
    std::vector<CommunicationStatistics> communication;
    CommunicationWork communication_eager, communication_graph;
    std::array<std::uint64_t, 3> compute_paths{};
    std::array<DurationCounter, 4> gpu_intervals{};
    std::optional<PhysicalMemorySample> physical_memory;
};

struct ScoringStatistics {
    std::uint64_t completed_jobs = 0, failed_jobs = 0, scored_tokens = 0;
    std::uint32_t active_jobs = 0;
    DurationCounter wall;
};

struct EngineSnapshot {
    MediaCacheSummary media;
    SnapshotStamp stamp;
    RuntimeStats runtime;
    ExecutionStatistics executions{};
    ScoringStatistics scoring;
    std::vector<RankSnapshot> ranks;
};

} // namespace ninfer
